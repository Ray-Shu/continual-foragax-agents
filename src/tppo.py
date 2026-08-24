import argparse
import logging
import os
import sys
import time
from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Experiment framework -- the SAME modules rtu_ppo.py imports. They resolve
# because this script lives in src/ alongside rtu_ppo.py (so `experiment`,
# `utils`, ... are importable as top-level packages). This is what lets tppo
# read experiments/*.json and write results in the shared format.
from experiment import ExperimentModel
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from PyExpUtils.results.tools import getParamsAsDict
from utils.checkpoint import Checkpoint
from utils.ml_instrumentation.Sampler import Mean
from utils.ml_instrumentation.utils import Last

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from algorithms.nn.tppo.transformer_ppo import TransformerPPO

sys.path.insert(0, os.path.abspath("/tmp/src"))
from foragax.registry import make  # noqa: E402

ACTION_DIM = 4

# ---------------------------------------------------------------------------
# Config -- just the knobs the core loop needs. Plain Python scalars; the jitted
# functions close over them as compile-time constants (so no config pytree to
# thread, unlike rtu_ppo's struct.dataclass).
# ---------------------------------------------------------------------------
class Config(NamedTuple):
    env_id: str
    aperture_size: int = 5
    env_kwargs: Tuple = ()
    seed: int = 0

    memory_len: int = 64 
    num_layers: int = 1
    d_hidden: int = 64
    d_keys: int = 32
    d_vals: int = 32
    d_ff: int = 128
    num_query_heads: int = 1  # attention heads for queries
    num_kv_heads: int = 1     # attention heads for keys/values
    activation: str = "relu"  
    gating: str = "residual"  
    gate_bias_init: float = 2.0  # look at gtrxl paper for more info 

    # ppo
    rollout_steps: int = 2048   # tokens collected per update (N); must be divisible by num_mini_batch
    total_steps: int = 10_000
    num_mini_batch: int = 4
    epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr_pi: float = 3e-4  # trunk (embed+blocks) + actor head
    lr_vf: float = 3e-4  # critic head only
    max_grad_norm: float = 0.5


# Rollout buffer entry.
class Transition(NamedTuple):
    action: jnp.ndarray      # a_t
    value: jnp.ndarray       # v(o_t), the transformer's value at step t
    reward: jnp.ndarray      # r_{t+1}, reward received after taking a_t
    log_prob: jnp.ndarray    # log pi(a_t | window_t)
    obs: Tuple[jnp.ndarray, ...]  # token_t = (obs_img, a_{t-1} one-hot, r_{t-1})


# Per-step foragax bookkeeping captured from env.step's info dict (+ env_state).
# These are the "cheap" logging columns rtu_ppo saves; the transformer needs none
# of them to learn -- they exist only so tppo's data/{idx}.npz matches rtu_ppo's.
class StepInfo(NamedTuple):
    pos: jnp.ndarray                 # agent position after the step, (pos_dim,)
    biome_id: jnp.ndarray            # scalar
    object_collected_id: jnp.ndarray  # scalar
    biome_regret: jnp.ndarray        # scalar
    biome_rank: jnp.ndarray          # scalar


# State carried across rollouts 
class RolloutState(NamedTuple):
    env_state: Any
    memory: Tuple[jnp.ndarray, ...]  # per-layer memory (1, memory_len, d_hidden)
    last_obs: jnp.ndarray            # next raw obs to turn into a token
    last_action: jnp.ndarray         # a_{t-1}
    last_reward: jnp.ndarray         # r_{t-1}
    timestep: jnp.ndarray


# Helpers
def make_token(obs_img, last_action, last_reward):
    """One transformer input token = (obs image, one-hot last action, last reward).

    Matches EmbeddingNet.__call__, which expects obs = (obs, last_action,
    last_reward) and flattens the image internally.
    """
    a = jax.nn.one_hot(last_action, ACTION_DIM)     # (ACTION_DIM,)
    r = jnp.reshape(last_reward, (1,))              # (1,)
    return (obs_img, a, r)


@jax.jit
def calculate_gae(traj_batch, last_val, gamma, gae_lambda):
    def _get_advantages(carry, transition):
        gae, next_value = carry
        value, reward = transition.value, transition.reward
        delta = reward + gamma * next_value - value
        gae = delta + gamma * gae_lambda * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


def parse_indices(index_specs: list[str], total: int | None = None) -> list[int]:
    indices = []
    for spec in index_specs:
        if ":" not in spec:
            indices.append(int(spec))
            continue

        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid index slice '{spec}', expected START:STOP")

        start_s, stop_s = parts
        start = int(start_s) if start_s else 0
        if stop_s:
            stop = int(stop_s)
        elif total is not None:
            stop = total
        else:
            raise ValueError(f"Open-ended index slice '{spec}' requires total runs")

        indices.extend(range(start, stop))

    return indices


# ---------------------------------------------------------------------------
# Training entry point. All the jitted functions are defined inside so they can
# close over the config, the env, and the network's (graphdef, rest) split as
# compile-time constants -- while `params` and `opt_state` are threaded as plain
# pytrees, exactly like rtu_ppo threads train_state.params.
#
# Returns (params, metrics) where `metrics` is a {name: np.ndarray} dict ready to
# hand straight to np.savez_compressed -- see main() for the save wiring.
# ---------------------------------------------------------------------------
def train(config: Config, num_updates: int | None = None):
    assert config.rollout_steps % config.num_mini_batch == 0, (
        "rollout_steps must be divisible by num_mini_batch"
    )
    N = config.rollout_steps
    mb_size = N // config.num_mini_batch
    if num_updates is None:
        num_updates = config.total_steps // config.rollout_steps + 1

    # ---- env ----
    env = make(config.env_id, aperture_size=config.aperture_size, **dict(config.env_kwargs))
    env_params = env.default_params
    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)

    if not hasattr(obs, "shape"):
        raise NotImplementedError(
            "Dict observations (image/hint) are not wired in this minimal script; "
            "use a plain-array foragax env or extend make_token()."
        )
    img_shape = obs.shape 

    # ---- network ----
    model = TransformerPPO(
        config.memory_len, config.d_hidden, config.d_keys, config.d_vals, config.d_ff,
        nnx.Rngs(config.seed), num_layers=config.num_layers,
        num_query_heads=config.num_query_heads, num_kv_heads=config.num_kv_heads,
        activation=config.activation, gating=config.gating,
        gate_bias_init=config.gate_bias_init,
    )
    # warm-up call: EmbeddingNet creates its Linear lazily on first call
    dummy_memory = model.init_memory(batch_size=1)
    dummy_token = make_token(jnp.zeros(img_shape), jnp.array(0, dtype=jnp.int32), jnp.array(0.0))
    dummy_obs = tuple(t[None, None] for t in dummy_token)
    _ = model(dummy_memory, dummy_obs)

    # Functional split: params are the trainable leaves (threaded + differentiated),
    # graphdef/rest are constant structure + non-Param state.
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
    print(f"[tppo] trainable params: {n_params:,}")
    print(f"[tppo] obs image shape: {img_shape}  | window token dim per step: "
          f"img{tuple(img_shape)} + act({ACTION_DIM}) + rew(1)")

    def forward(params, memory, obs, read_from=None):
        """Batched forward: window is a tuple of (B, M, .) arrays.

        read_from=None  -> (pi, value) at the LAST position only. Rollout path.
        read_from=prefix-> (pi, value) at every position from `prefix` on, i.e.
                           one decision per rollout transition. Update path.
        """
        model = nnx.merge(graphdef, params, rest)
        return model(memory, obs, read_from=read_from)

    def batched(seq):
        """Add a leading batch axis to a single (M, .) sequence -> (1, M, .)."""
        return tuple(s[None] for s in seq)

    # ---- optimiser (plain optax, threaded like rtu_ppo's tx) ----
    # Split by param path into "vf" (critic head) vs "pi" (everything else --
    # the actor head plus the shared embed/blocks trunk), same convention
    # rtu_ppo uses, so optimizer_actor/optimizer_critic give independent LRs.
    def make_label_tree(params):
        paths_leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
        labels = [
            "vf" if "critic" in jax.tree_util.keystr(path) else "pi"
            for path, _ in paths_leaves
        ]
        return jax.tree_util.tree_unflatten(treedef, labels)

    labels = make_label_tree(params)
    tx = optax.partition(
        {
            "pi": optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr_pi),
            ),
            "vf": optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr_vf),
            ),
        },
        labels,
    )
    opt_state = tx.init(params)

    # ---------------- rollout (jitted) ----------------
    @jax.jit
    def collect_rollout(params, state: RolloutState, rng):
        def _step(carry, _):
            env_state, memory, last_obs, last_action, last_reward, t, rng = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)

            token = make_token(last_obs, last_action, last_reward)  # (o_t, a_{t-1}, r_{t-1})
            token_b = tuple(tk[None, None] for tk in token)                         
            pi, value, new_memory = forward(params, memory, token_b)            # (1, T, .) -> B=1
            action_b = pi.sample(seed=a_rng)                        # (1,)
            log_prob_b = pi.log_prob(action_b)                      # (1,)
            action = jnp.squeeze(action_b, 0)                       # scalar for env.step
            log_prob = jnp.squeeze(log_prob_b, 0)

            obs2, env_state, reward, done, info = env.step(
                s_rng, env_state, action, env_params
            )
            transition = Transition(
                action=action,
                value=jnp.squeeze(value),
                reward=reward,
                log_prob=log_prob,
                obs=token,
            )
          
            step_info = StepInfo(
                pos=env_state.pos,
                biome_id=info["biome_id"],
                object_collected_id=info["object_collected_id"],
                biome_regret=info["biome_regret"],
                biome_rank=info["biome_rank"],
            )
            carry = (env_state, new_memory, obs2, action, reward, t + 1, rng)
            return carry, (transition, step_info)

        carry = (
            state.env_state, state.memory, state.last_obs,
            state.last_action, state.last_reward, state.timestep, rng,
        )

        trajs, step_info_chunks, memory_snapshots = [], [], [] 
        for _ in range(config.num_mini_batch): 
            memory_snapshots.append(carry[1])
            carry, (traj_chunk, info_chunk) = jax.lax.scan(
                _step, carry, None, length=mb_size

            )
            trajs.append(traj_chunk)
            step_info_chunks.append(info_chunk)

        env_state, memory, last_obs, last_action, last_reward, t, rng = carry 
        traj = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), * trajs)
        step_infos = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *step_info_chunks) 
        memory_snapshots = tuple(
            jnp.stack(layer_snaps, axis=0) for layer_snaps in zip(*memory_snapshots)
        )

        boot_token = make_token(last_obs, last_action, last_reward)
        boot_token_b = tuple(tk[None, None] for tk in boot_token)
        _, last_val, _ = forward(params, memory, boot_token_b)

        new_state = RolloutState(
            env_state = env_state, memory=memory, last_obs=last_obs,
            last_action=last_action, last_reward=last_reward, timestep=t
        )

        return new_state, jnp.squeeze(last_val), traj, step_infos, memory_snapshots


    # ---------------- loss + update (jitted) ----------------
    def loss_fn(params,memory, seq, actions, old_log_prob, old_value, adv, targets):
        # memory: this minibatch's starting memory (per-layer tuple)
        # seq: tuple of (mb_size, .) 
        seq_b = batched(seq)                                  # (1, prefix+mb, .)
        pi, values, _ = forward(params, memory, seq_b, read_from=0)  # pi:(1,mb) values:(1,mb,1)
        log_prob = pi.log_prob(actions[None])[0]              # (mb,)
        values = values[0, :, 0]                              # (mb,)
        entropy = pi.entropy().mean()

        # value loss (clipped, as in rtu_ppo)
        v_clipped = old_value + jnp.clip(values - old_value, -config.clip_eps, config.clip_eps)
        value_loss = 0.5 * jnp.maximum(
            jnp.square(values - targets), jnp.square(v_clipped - targets)
        ).mean()

        # policy loss (clipped surrogate)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = jnp.exp(log_prob - old_log_prob)
        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
        pg_loss = -jnp.minimum(pg1, pg2).mean()

        total = pg_loss + config.vf_coef * value_loss - config.ent_coef * entropy
        return total, (value_loss, pg_loss, entropy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def ppo_update(params, opt_state, traj_obs, memory_snapshots, data, rng):
        """
        traj_obs        : tuple of (N, .) -- this rollout's tokens.
        memory_snapshots: tuple of (num_mini_batch, 1, memory_len, d_hidden) per layer 
        data          : (actions, old_log_prob, old_value, adv, targets), leading axis N.
        """
        def _epoch(carry, _):
            params, opt_state, rng = carry
            rng, perm_rng = jax.random.split(rng)

            perm = jax.random.permutation(perm_rng, config.num_mini_batch)

            def _minibatch(carry, c):
                params, opt_state = carry
                start = c * mb_size
                seq = jax.tree_util.tree_map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, start, mb_size, axis=0),
                    traj_obs
                )
                mem_c = jax.tree_util.tree_map(lambda m: m[c], memory_snapshots)
                batch = jax.tree_util.tree_map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, start, mb_size, axis=0),
                    data,
                )
                (loss, aux), grads = grad_fn(params, mem_c, seq, *batch)
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), (loss, aux)

            (params, opt_state), out = jax.lax.scan(
                _minibatch, (params, opt_state), perm
            )
            return (params, opt_state, rng), out

        (params, opt_state, rng), out = jax.lax.scan(
            _epoch, (params, opt_state, rng), None, length=config.epochs
        )
        return params, opt_state, out

    # ---------------- outer loop (plain Python, one jit call per phase) ----------------
    state = RolloutState(
        env_state=env_state,
        memory=model.init_memory(batch_size=1), 
        last_obs=obs,
        last_action=jnp.array(0, dtype=jnp.int32),
        last_reward=jnp.array(0.0, dtype=jnp.float32),
        timestep=jnp.array(0, dtype=jnp.int32),
    )

    # Per-run metric history, host-side numpy, assembled into the npz below.
    # Per-env-step columns (concatenated across updates) and per-update columns.
    rewards_hist, pos_hist = [], []
    biome_id_hist, obj_hist, regret_hist, rank_hist = [], [], [], []
    total_loss_hist, value_loss_hist, policy_loss_hist, entropy_hist = [], [], [], []

    for it in range(num_updates):
        rng, c_rng, u_rng = jax.random.split(rng, 3)


        # --- collect (timed; first iter includes JIT compile) ---
        t0 = time.perf_counter()
        new_state, last_val, traj, step_infos, memory_snapshots = collect_rollout(params, state, c_rng)
        jax.block_until_ready((last_val, traj.reward))
        t_collect = time.perf_counter() - t0

        # --- advantages ---
        adv, targets = calculate_gae(traj, last_val, config.gamma, config.gae_lambda)

        data = (traj.action, traj.log_prob, traj.value, adv, targets)

        t0 = time.perf_counter()
        params, opt_state, out = ppo_update(params, opt_state, traj.obs, memory_snapshots, data, u_rng)
        jax.block_until_ready(out)
        t_update = time.perf_counter() - t0

        state = new_state
        (loss, (v_loss, pg_loss, ent)) = out

        # --- accumulate logging columns (see StepInfo / rtu_ppo save block) ---
        rewards_hist.append(np.asarray(traj.reward))          # (N,)
        pos_hist.append(np.asarray(step_infos.pos))            # (N, pos_dim)
        biome_id_hist.append(np.asarray(step_infos.biome_id))
        obj_hist.append(np.asarray(step_infos.object_collected_id))
        regret_hist.append(np.asarray(step_infos.biome_regret))
        rank_hist.append(np.asarray(step_infos.biome_rank))
        # One scalar per update: mean over the (epochs, num_mini_batch) grid.
        total_loss_hist.append(float(loss.mean()))
        value_loss_hist.append(float(v_loss.mean()))
        policy_loss_hist.append(float(pg_loss.mean()))
        entropy_hist.append(float(ent.mean()))

        print(
            f"[{it:03d}] reward/step={float(traj.reward.mean()):+.4f}  "
            f"loss={float(loss.mean()):.3f}  v={float(v_loss.mean()):.3f}  "
            f"pg={float(pg_loss.mean()):.3f}  ent={float(ent.mean()):.3f}  "
            f"| collect={t_collect*1e3:6.1f}ms  update={t_update*1e3:6.1f}ms"
        )

    # Assemble the npz payload with the SAME shapes rtu_ppo saves: per-step
    # columns flattened to (num_updates*N, ...), per-update columns (num_updates,).
    metrics = {
        "rewards": np.concatenate(rewards_hist).astype(np.float32),
        "pos": np.concatenate(pos_hist, axis=0),
        "biome_id": np.concatenate(biome_id_hist),
        "object_collected_id": np.concatenate(obj_hist),
        "biome_regret": np.concatenate(regret_hist),
        "biome_rank": np.concatenate(rank_hist),
        "total_loss": np.asarray(total_loss_hist, dtype=np.float32),
        "value_loss": np.asarray(value_loss_hist, dtype=np.float32),
        "policy_loss": np.asarray(policy_loss_hist, dtype=np.float32),
        "entropy": np.asarray(entropy_hist, dtype=np.float32),
    }
    return params, metrics


def _build_config(exp: ExperimentModel, hypers: dict, seed: int) -> Config:
    """Map a resolved `metaParameters` dict -> tppo Config.

    PPO / env fields read the same JSON keys rtu_ppo reads. The transformer-only
    knobs (memory_len, num_layers, d_keys, d_vals, d_ff, num_query_heads, num_kv_heads,
    activation, gating, gate_bias_init) come from the
    `representation` block, each falling back to the Config default when absent --
    so an ActorCriticMLP.json (which has none of them) still runs, and a
    TransformerPPO.json can pin them.
    """
    d = Config._field_defaults
    env = hypers["environment"]
    rep = hypers.get("representation", {})
    # Everything under `environment` except the three handled explicitly becomes
    # a make() kwarg (e.g. observation_type) -- same rule as rtu_ppo.
    env_kwargs = tuple(sorted(
        (k, v) for k, v in env.items()
        if k not in ("aperture_size", "env_id", "render_mode") and v is not None
    ))
    return Config(
        env_id=env["env_id"],
        aperture_size=int(env["aperture_size"]),
        env_kwargs=env_kwargs,
        seed=seed,
        memory_len=int(rep.get("memory_len", d["memory_len"])),
        num_layers=int(rep.get("num_layers", d["num_layers"])),
        d_hidden=int(rep.get("d_hidden", d["d_hidden"])),
        d_keys=int(rep.get("d_keys", d["d_keys"])),
        d_vals=int(rep.get("d_vals", d["d_vals"])),
        d_ff=int(rep.get("d_ff", d["d_ff"])),
        num_query_heads=int(rep.get("num_query_heads", d["num_query_heads"])),
        num_kv_heads=int(rep.get("num_kv_heads", d["num_kv_heads"])),
        activation=str(rep.get("activation", d["activation"])),
        gating=str(rep.get("gating", d["gating"])),
        gate_bias_init=float(rep.get("gate_bias_init", d["gate_bias_init"])),
        rollout_steps=int(hypers["rollout_steps"]),
        total_steps=int(exp.total_steps),
        num_mini_batch=int(hypers["num_mini_batch"]),
        epochs=int(hypers["epochs"]),
        gamma=float(hypers["gamma"]),
        gae_lambda=float(hypers["gae_lambda"]),
        clip_eps=float(hypers["clip_eps"]),
        vf_coef=float(hypers["vf_coef"]),
        ent_coef=float(hypers["entropy_coef"]),
        lr_pi=float(hypers["optimizer_actor"]["alpha"]),
        lr_vf=float(
            hypers["optimizer_critic"].get(
                "alpha",
                hypers["optimizer_critic"].get("lr_scale", float("nan"))
                * hypers["optimizer_actor"]["alpha"],
            )
        ),
        max_grad_norm=float(hypers["max_grad_norm"]),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the sliding-window Transformer-PPO agent from an "
                    "experiments/*.json config (same -e/-i CLI as rtu_ppo.py)."
    )
    parser.add_argument("-e", "--exp", type=str, required=True,
                        help="path to an experiment .json (e.g. "
                             "experiments/.../TransformerPPO.json)")
    parser.add_argument("-i", "--idxs", nargs="+", type=str, required=True,
                        help="run indices: '0', '0 2 4', or a slice '0:5'")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
    parser.add_argument("--silent", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="cap the number of PPO updates (overrides the "
                             "total_steps-derived count); handy for smoke tests")
    args = parser.parse_args()

    if not args.gpu:
        jax.config.update("jax_platform_name", "cpu")

    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logger = logging.getLogger("tppo")
    if not args.silent:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

    exp = ExperimentModel.load(args.exp)
    if not exp.agent.startswith("TransformerPPO"):
        logger.warning(
            "experiment agent is '%s', which is not a TransformerPPO variant; "
            "running the transformer anyway (any knobs absent from the "
            "representation block fall back to Config defaults).",
            exp.agent,
        )
    try:
        indices = parse_indices(args.idxs, exp.numPermutations())
    except ValueError as e:
        parser.error(str(e))

    for idx in indices:
        # --- checkpoint + collector lifecycle (mirrors rtu_ppo) ---
        chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
        chk.load_if_exists()
        collector = chk.build(
            "collector",
            lambda: Collector(
                config={
                    "ewm_reward": Pipe(
                        MovingAverage(0.999),
                        Subsample(max(exp.total_steps // 1000, 1)),
                    ),
                    "mean_ewm_reward": Last(MovingAverage(0.999), Mean()),
                },
                default=Ignore(),
            ),
        )
        collector.set_experiment_id(idx)

        # --- resolve this index's hypers -> Config ---
        hypers = exp.get_hypers(idx)
        seed = exp.getRun(idx) + hypers.get("seed_offset", 0)
        num_updates = (
            int(hypers["num_updates"]) if "num_updates" in hypers
            else (exp.total_steps // int(hypers["rollout_steps"]) + 1)
        )
        if args.max_steps is not None:
            num_updates = args.max_steps
        config = _build_config(exp, hypers, seed)
        logger.info(
            "idx=%d seed=%d env=%s memory_len=%d num_layers=%d rollout=%d num_updates=%d",
            idx, seed, config.env_id, config.memory_len, config.num_layers,
            config.rollout_steps, num_updates,
        )

        # --- train ---
        _params, metrics = train(config, num_updates=num_updates)

        # --- save results.db + data/{idx}.npz (identical layout to rtu_ppo) ---
        # The reward/loss/info time series go in the npz. The Collector is reset
        # (per-step population is intentionally skipped, exactly as in rtu_ppo,
        # which found it too slow) and merged so results.db carries the run's
        # metadata for the plotting pipeline to discover.
        collector.reset()
        context = exp.buildSaveContext(idx, base=args.save_path)
        save_path = context.resolve("results.db")
        data_path = context.resolve(f"data/{idx}.npz")
        context.ensureExists(data_path, is_file=True)
        np.savez_compressed(data_path, **metrics)

        meta = getParamsAsDict(exp, idx)
        meta |= {"seed": exp.getRun(idx)}
        attach_metadata(save_path, idx, meta)

        collector.merge(save_path)
        collector.close()
        chk.delete()
        logger.info("idx=%d saved -> %s", idx, data_path)


if __name__ == "__main__":
    main()
