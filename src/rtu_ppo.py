# Modified from esraaelelimy/continuing_ppo

# Ensure third-party libraries that expect older JAX internals can import.
# This sets a small compatibility alias if needed before importing distrax/tfp.
import argparse
import utils.jax_compat  # noqa: F401
import socket
import time
import logging
import os
import sys
import zlib
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct, traverse_util
from flax.training.train_state import TrainState
from gymnasium.utils.save_video import save_video
from jax.tree_util import tree_map
from jax_tqdm.base import PBar
from jax_tqdm.scan_pbar import scan_tqdm
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from PyExpUtils.results.tools import getParamsAsDict

from algorithms.nn.ACConv import ActorCriticConv
from algorithms.nn.ACMLP import ActorCriticMLP
from algorithms.nn.RealTimeACConv import RealTimeActorCriticConv
from algorithms.nn.RealTimeACConvHint import RealTimeActorCriticConvHint
from algorithms.nn.RealTimeACConvHintRTU import RealTimeActorCriticConvHintRTU
from algorithms.nn.RealTimeACConvPooling import RealTimeActorCriticConvPooling
from algorithms.nn.RealTimeACMLP import RealTimeActorCriticMLP
from algorithms.nn.RealTimeACMLPMulti import RealTimeActorCriticMLPMulti
from algorithms.PPORegistry import getAgent
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.ml_instrumentation.Sampler import Mean
from utils.ml_instrumentation.utils import Last
from utils.preempt import TimeoutHandler

sys.path.insert(0, os.path.abspath("/tmp/src"))
from foragax.registry import make

PERIOD = 182500


def _keypath_crc32(path) -> int:
    path_str = "/".join(str(getattr(entry, "key", entry)) for entry in path)
    return zlib.crc32(path_str.encode("utf-8"))


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


def _crossed_interval(start_step, end_step, interval):
    return (end_step > 0) & ((end_step // interval) > (start_step // interval))


@struct.dataclass
class LogEnvState:
    returned_returns: float
    timestep: int
    frames: Any


@struct.dataclass
class TrainConfig:
    # ---- STATIC (uniform across vmapped runs) ----
    d_hidden: int = struct.field(pytree_node=False)
    hidden_size: int = struct.field(pytree_node=False)
    agent_type: str = struct.field(pytree_node=False)
    rollout_steps: int = struct.field(pytree_node=False)
    epochs: int = struct.field(pytree_node=False)
    num_mini_batch: int = struct.field(pytree_node=False)
    gradient_clipping: bool = struct.field(pytree_node=False)
    num_updates: int = struct.field(pytree_node=False)
    env_id: str = struct.field(pytree_node=False)
    aperture_size: int = struct.field(pytree_node=False)
    render_mode: str = struct.field(pytree_node=False)
    env_kwargs: Any = struct.field(pytree_node=False)
    use_sinusoidal_encoding: bool = struct.field(pytree_node=False)
    use_reward_trace: bool = struct.field(pytree_node=False)
    use_hint_trace: bool = struct.field(pytree_node=False)
    use_layernorm: bool = struct.field(pytree_node=False)
    conv: str = struct.field(pytree_node=False)
    allocate_frames: bool = struct.field(pytree_node=False)
    video_length: int = struct.field(pytree_node=False)
    use_l2_init: bool = struct.field(pytree_node=False)
    use_spectral_reg: bool = struct.field(pytree_node=False)
    use_reset: bool = struct.field(pytree_node=False)
    use_shrink_and_perturb: bool = struct.field(pytree_node=False)
    # ---- DYNAMIC (may vary per idx; arithmetic only) ----
    max_grad_norm: float
    l2_reg_pi: float
    l2_reg_vf: float
    alpha_pi: float
    alpha_vf: float
    adam_eps_pi: float
    adam_eps_vf: float
    adam_b1_pi: float
    adam_b2_pi: float
    adam_b1_vf: float
    adam_b2_vf: float

    sparsity: float
    spectral_radius: float

    id: int
    reward_trace_decay: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    vf_coef: float
    entropy_coef: float
    lambda_l2_init_pi: float = 0.0
    lambda_l2_init_vf: float = 0.0
    lambda_spectral_pi: float = 0.0
    lambda_spectral_vf: float = 0.0
    freeze_steps: int = -1
    reset_interval: int = -1
    sp_interval: int = -1
    shrink_factor: float = 1.0
    noise_scale: float = 0.0

class GymnaxEnvState(struct.PyTreeNode):
    to_render: bool = struct.field(pytree_node=True)
    cond_render: Callable = struct.field(pytree_node=False)
    env_step: Callable = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=True)
    env_state: Any = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, env_step, env_params, env_state, **kwargs):
        """Creates a new instance"""
        return cls(
            env_step=env_step,
            env_params=env_params,
            env_state=env_state,
            **kwargs,
        )


class Transition(NamedTuple):
    action: jnp.ndarray  # a_t
    value: jnp.ndarray  # v(o_t)
    reward: jnp.ndarray  # r[t+1]
    log_prob: jnp.ndarray
    obs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]  # o_t, a_{t-1}, r_{t-1}
    info: jnp.ndarray


class Interaction(NamedTuple):
    a: int
    r: bool


@jax.jit
def calculate_gae(traj_batch, last_val, gamma, gae_lambda):
    def _get_advantages(carry, transition):
        gae, next_value = carry
        value, reward = (
            transition.value,
            transition.reward,
        )
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


@jax.jit
def calculate_average_reward_gae(traj_batch, last_val, gamma, gae_lambda):
    sample_avg_reward = jnp.mean(traj_batch.reward)  # r_\pi

    def _get_advantages(next_value, transition):
        value, reward = (
            transition.value,
            transition.reward,
        )
        gae = reward - sample_avg_reward + next_value - value
        return value, gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        last_val,
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


@partial(jax.jit, static_argnums=(1, 9, 12))
def loss_fn(
    params, agent_fn, traj_batch, gae, targets, init_hstate, clip_eps, vf_coef, ent_coef,
    use_l2_init=False, initial_params=None, l2_init_multipliers=None,
    use_spectral_reg=False, spectral_reg_multipliers=None,
):
    rnn_in = traj_batch.obs
    _, pi, value = agent_fn(params, init_hstate, rnn_in)
    log_prob = pi.log_prob(traj_batch.action)
    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
        -clip_eps, clip_eps
    )
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    total_loss = loss_actor + vf_coef * value_loss - ent_coef * entropy

    # L2-to-init regularisation (guard: compiled away when use_l2_init is False)
    if use_l2_init:
        # optax.l2_loss(a, b) = 0.5 * (a - b)^2 per element
        l2_init_loss = jax.tree_util.tree_map(
            lambda m, p, p0: m * optax.l2_loss(p, p0).sum(),
            l2_init_multipliers, params, initial_params,
        )
        total_loss = total_loss + jax.tree_util.tree_reduce(
            lambda a, b: a + b, l2_init_loss
        )

    # Spectral regularisation (Lyle, Rowland, Dabney & Gal, 2024; adapted
    # following Machado's group work on loss of plasticity).
    #
    # For each layer l with weight W_l and bias b_l the regulariser is:
    #   R(θ_l) = (σ₁(W_l)^k − 1)² + ‖b_l‖₂^(2k)
    # with k = 2 (paper default).
    #
    # Special cases handled at trace time via parameter name:
    #   • Dense/Conv kernel  → (σ₁^k − 1)²   (spectral norm towards 1)
    #   • Bias               → ‖b‖₂^(2k)      (towards 0)
    #   • LayerNorm scale    → Σ(γ_i − 1)²    (element-wise towards 1)
    #   • Conv 4-D tensors   → reshape to (d_out, k·k·d_in) then σ₁
    #
    # σ₁ is estimated with a single power-iteration step (Yoshida & Miyato,
    # 2017) which the paper finds sufficient.  The compile-time flag
    # `use_spectral_reg` causes this entire block to be compiled away when
    # spectral regularisation is not requested.
    if use_spectral_reg:
        _SR_K = 2  # exponent k from the paper

        def _power_iteration_sigma1(w_2d, num_iters=1):
            """Estimate σ₁(w_2d) via power iteration (1 step by default)."""
            u = jnp.ones((w_2d.shape[0],), dtype=w_2d.dtype)
            u = u / (jnp.linalg.norm(u) + 1e-12)
            def _step(u, _):
                v = w_2d.T @ u
                v = v / (jnp.linalg.norm(v) + 1e-12)
                u_new = w_2d @ v
                u_new = u_new / (jnp.linalg.norm(u_new) + 1e-12)
                return u_new, v
            u, vs = jax.lax.scan(_step, u, None, length=num_iters)
            v = vs[-1]
            sigma = u @ w_2d @ v
            return sigma

        def _spectral_leaf(path, multiplier, param):
            """Per-leaf spectral regularisation loss (dispatched at trace time)."""
            # Identify the leaf name from its path key
            leaf_name = (
                path[-1].key.lower()
                if hasattr(path[-1], "key")
                else str(path[-1]).lower()
            )

            # --- LayerNorm / normalisation scale parameters ---
            # Paper: "regularize each weight towards 1"
            if leaf_name in ("scale", "gamma"):
                return multiplier * jnp.sum(jnp.square(param - 1.0))

            # --- Bias / additive parameters ---
            # Paper: ‖b‖₂^(2k)
            if leaf_name in ("bias", "beta", "offset") or param.ndim == 1:
                return multiplier * jnp.linalg.norm(param) ** (2 * _SR_K)

            # --- Convolutional kernels (4-D) ---
            # Paper (Appendix A.7): reshape to (d_out, k·k·d_in)
            # Flax Conv default layout: (spatial..., d_in, d_out)
            if param.ndim == 4:
                # (h, w, d_in, d_out) → (d_out, h*w*d_in)
                d_out = param.shape[-1]
                w_2d = jnp.transpose(param, (3, 0, 1, 2)).reshape((d_out, -1))
                sigma = _power_iteration_sigma1(w_2d)
                return multiplier * jnp.square(sigma ** _SR_K - 1.0)

            # --- Dense / multiplicative weight matrices (2-D) ---
            # Paper: (σ₁(W)^k − 1)²
            if param.ndim == 2:
                sigma = _power_iteration_sigma1(param)
                return multiplier * jnp.square(sigma ** _SR_K - 1.0)

            # Anything else (scalars, etc.) – skip
            return jnp.zeros((), dtype=param.dtype)

        spectral_losses = jax.tree_util.tree_map_with_path(
            _spectral_leaf, spectral_reg_multipliers, params,
        )
        total_loss = total_loss + jax.tree_util.tree_reduce(
            lambda a, b: a + b, spectral_losses
        )

    return total_loss, (value_loss, loss_actor, entropy)


@jax.jit
def agent_step(last_obs, train_state, rng, hstate):
    last_obs, last_action_encoded, last_reward, sine, cosine, reward_trace = last_obs
    rnn_in = (
        jnp.expand_dims(last_obs, 0),
        jnp.expand_dims(last_action_encoded, 0),
        jnp.expand_dims(last_reward, 0),
        jnp.expand_dims(sine, 0),
        jnp.expand_dims(cosine, 0),
        jnp.expand_dims(reward_trace, 0),
    )
    last_hidden, pi, value = train_state.apply_fn(train_state.params, hstate, rnn_in)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value, last_hidden


def env_step(runner_state, _):
    (
        train_state,
        gymnax_state,
        log_env_state,
        config,
        last_obs,
        last_action,
        last_reward,
        reward_trace,
        hint_trace,
        rng,
        hstate,
    ) = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    action_encoded = jnp.zeros((4,))
    action_encoded = action_encoded.at[last_action].set(1)
    last_reward_encoded = jnp.expand_dims(last_reward, 0)

    if isinstance(last_obs, Mapping):
        obs_img = last_obs["image"]
        hint = last_obs["hint"]
        if config.use_hint_trace:
            hint_trace = (
                config.reward_trace_decay * hint_trace
                + (1.0 - config.reward_trace_decay) * hint
            )
            hint = hint_trace
        last_reward_encoded = jnp.concatenate((last_reward_encoded, hint), axis=-1)
    else:
        obs_img = last_obs

    sine = jnp.expand_dims(jnp.sin(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
    cosine = jnp.expand_dims(jnp.cos(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
    reward_trace = (
        config.reward_trace_decay * reward_trace
        + (1.0 - config.reward_trace_decay) * last_reward
    )
    reward_trace_encoded = jnp.expand_dims(reward_trace, 0)
    last_obs_encoded = (
        obs_img,
        action_encoded,
        last_reward_encoded,
        sine,
        cosine,
        reward_trace_encoded,
    )

    action, log_prob, value, last_hidden = agent_step(
        last_obs_encoded, train_state, _rng, hstate
    )
    # STEP ENV
    obs, env_state, reward, done, info = gymnax_state.env_step(
        _rng, gymnax_state.env_state, action.squeeze(), gymnax_state.env_params
    )
    step = log_env_state.timestep + 1
    new_return = 0.999 * log_env_state.returned_returns + (1.0 - 0.999) * (reward)

    frame = gymnax_state.cond_render(gymnax_state.to_render, gymnax_state.env_state)

    log_env_state = LogEnvState(
        returned_returns=new_return, timestep=step, frames=log_env_state.frames
    )

    info["reward"] = reward
    info["moving_average"] = new_return
    info["timestep"] = log_env_state.timestep
    info["pos"] = env_state.pos
    info["frame"] = frame

    ### Create transition
    transition = Transition(
        action.squeeze(),
        value.squeeze(),
        reward,
        log_prob.squeeze(),
        last_obs_encoded,
        info,
    )
    ### Update runner state
    gymnax_state = GymnaxEnvState.create(
        to_render=gymnax_state.to_render,
        cond_render=gymnax_state.cond_render,
        env_step=gymnax_state.env_step,
        env_params=gymnax_state.env_params,
        env_state=env_state,
    )
    runner_state = (
        train_state,
        gymnax_state,
        log_env_state,
        config,
        obs,
        action.squeeze(),
        reward,
        reward_trace,
        hint_trace,
        rng,
        last_hidden,
    )
    return runner_state, (transition, hstate)


@jax.jit
def update_minbatch(carry_in, batch_info):
    train_state, config, initial_params, l2_init_multipliers, spectral_reg_multipliers = carry_in
    minibatch, init_hstate = batch_info
    # minibatch: (seq_len,minibatch_size, _)
    # init_hstate: (1, d_hidden)
    traj_batch, advantages, targets = minibatch
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    total_loss, grads = grad_fn(
        train_state.params,
        train_state.apply_fn,
        traj_batch,
        advantages,
        targets,
        init_hstate,
        config.clip_eps,
        config.vf_coef,
        config.entropy_coef,
        config.use_l2_init,
        initial_params,
        l2_init_multipliers,
        config.use_spectral_reg,
        spectral_reg_multipliers,
    )
    train_state = train_state.apply_gradients(grads=grads)
    return (train_state, config, initial_params, l2_init_multipliers, spectral_reg_multipliers), total_loss


"""
Batch shape = (num_steps, _)
Divide the batch into n minibatches
each minibatch has the shape of (seq_len, minibatch_size, _)
minibatch_size = num_steps//n*seq_len

1. re-run the network through the batch and store hiddens states for positions (0,seq_len,2*seq_len,...)
2. Divide num_steps into sequences of length seq_len : number of sequences = num_steps//seq_len
3. Divide the sequences into n minibatches
4. shuffle the minibatches
output shape = (num_minibatches, seq_len, minibatch_size, _)
"""


@jax.jit
def create_minibaches(config: TrainConfig, hstate_batch, batch, rng, train_state):
    batch_hstate = jax.tree_util.tree_map(
        lambda y: jnp.squeeze(y, axis=1), hstate_batch
    )
    traj_batch, advantages, targets = batch
    batch = (batch_hstate, traj_batch, advantages, targets)

    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, config.rollout_steps)
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )

    minibatch_size = config.rollout_steps // config.num_mini_batch
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: x.reshape(
            (
                config.num_mini_batch,
                minibatch_size,
            )
            + x.shape[1:]
        ),
        shuffled_batch,
    )

    batch_hstate, traj_batch, advantages, targets = shuffled_batch
    minibatches_info = ((traj_batch, advantages, targets), batch_hstate)
    return minibatches_info, rng


@jax.jit
def update_epoch(update_state, unused):
    (
        train_state,
        init_hstate,
        traj_batch,
        hstate_batch,
        advantages,
        targets,
        rng,
        config,
        initial_params,
        l2_init_multipliers,
        spectral_reg_multipliers,
    ) = update_state
    batch = (traj_batch, advantages, targets)
    minibatches_info, rng = create_minibaches(
        config, hstate_batch, batch, rng, train_state
    )
    carry_in = (train_state, config, initial_params, l2_init_multipliers, spectral_reg_multipliers)
    carry_out, total_loss = jax.lax.scan(update_minbatch, carry_in, minibatches_info)
    train_state = carry_out[0]
    update_state = (
        train_state,
        init_hstate,
        traj_batch,
        hstate_batch,
        advantages,
        targets,
        rng,
        config,
        initial_params,
        l2_init_multipliers,
        spectral_reg_multipliers,
    )
    return update_state, total_loss


@jax.jit
def update_step(update_state):
    (
        train_state,
        init_hstate,
        traj_batch,
        hstate_batch,
        advantages,
        targets,
        rng,
        config,
        initial_params,
        l2_init_multipliers,
        spectral_reg_multipliers,
    ) = update_state
    update_state, loss_info = jax.lax.scan(
        update_epoch, update_state, None, config.epochs
    )
    ## Update runner state
    train_state = update_state[0]
    rng = update_state[6]
    return (train_state, rng), loss_info


def experiment(rng, config: TrainConfig):
    kwargs = dict(config.env_kwargs)

    print(f"Creating env {config.env_id} with aperture size {config.aperture_size} and kwargs {kwargs}")

    env = make(config.env_id, aperture_size=config.aperture_size, **kwargs)

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env.default_params)

    def real_render(env_state):
        return env.render(env_state, None, render_mode=config.render_mode).astype(
            jnp.uint8
        )

    render_shape = real_render(env_state).shape

    ### Initialize the environment states
    if config.allocate_frames:
        updates_per_video = (
            config.video_length + config.rollout_steps - 1
        ) // config.rollout_steps
        frames = jnp.zeros(
            (
                updates_per_video * config.rollout_steps,
                *render_shape,
            ),
            dtype=jnp.uint8,
        )
    else:
        frames = jnp.zeros((0, *render_shape), dtype=jnp.uint8)
    log_env_state = LogEnvState(returned_returns=0, timestep=0, frames=frames)

    def void_render(env_state):
        return jnp.zeros(render_shape, dtype=jnp.uint8)

    def render(cond, env_state):
        return jax.lax.cond(cond, real_render, void_render, env_state)

    gymnax_state = GymnaxEnvState.create(
        to_render=False,
        cond_render=render,
        env_step=env.step,
        env_params=env.default_params,
        env_state=env_state,
    )
    action_dim = 4

    _agent_class = getAgent(config.agent_type)
    agent = _agent_class

    kwargs = {}
    if config.sparsity is not None:
        kwargs["sparsity"] = config.sparsity
    if config.spectral_radius is not None:
        kwargs["spectral_radius"] = config.spectral_radius
    if _agent_class in (
        ActorCriticConv,
        RealTimeActorCriticConv,
        RealTimeActorCriticConvPooling,
        RealTimeActorCriticConvHint,
        RealTimeActorCriticConvHintRTU,
    ):
        kwargs["conv"] = config.conv

    # Create and initialize the network.
    network = agent(
        action_dim=action_dim,
        activation="tanh",
        hidden_size=config.hidden_size,
        d_hidden=config.d_hidden,
        cont=False,
        use_sinusoidal_encoding=config.use_sinusoidal_encoding,
        use_reward_trace=config.use_reward_trace,
        use_layernorm=config.use_layernorm,
        **kwargs,
    )

    rng, _rng = jax.random.split(rng)

    if isinstance(obs, Mapping):
        obs_img_shape = obs["image"].shape
        hint_shape = (1 + obs["hint"].shape[-1],)
        hint_dim = obs["hint"].shape[-1]
    else:
        obs_img_shape = obs.shape
        hint_shape = (1,)
        hint_dim = 1  # placeholder; not used for non-hint envs

    init_x = (
        jnp.zeros((1, *obs_img_shape)),
        jnp.zeros((1, action_dim)),
        jnp.zeros((1, *hint_shape)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )

    _is_conv_rtu = _agent_class in (
        RealTimeActorCriticConv,
        RealTimeActorCriticConvPooling,
        RealTimeActorCriticConvHint,
    )
    _is_conv_hint_rtu = _agent_class is RealTimeActorCriticConvHintRTU
    _is_mlp_rtu = _agent_class in (RealTimeActorCriticMLP, RealTimeActorCriticMLPMulti, ActorCriticMLP)
    if _is_conv_rtu:
        # RTU receives hidden_size-wide embedding; action/reward/hint folded in before the Dense
        d_input = config.hidden_size
    elif _is_conv_hint_rtu:
        # No main RTU — d_input is ignored by initialize_memory, pass hint input size
        d_input = hint_dim
    elif _is_mlp_rtu:
        # RTU receives [Dense(hidden_size), action, last_reward+hint, ...]
        # hint_shape[0] = 1 + hint_dim (accounts for reward + hint)
        d_input = config.hidden_size + action_dim + hint_shape[0]
        if config.use_sinusoidal_encoding:
            d_input += 2
        if config.use_reward_trace:
            d_input += 1
    else:
        d_input = config.hidden_size
    if _is_conv_hint_rtu:
        init_hstate = agent.initialize_memory(1, config.d_hidden, hint_dim)
    else:
        init_hstate = agent.initialize_memory(1, config.d_hidden, d_input)
    network_params = network.init(_rng, init_hstate, init_x)

    def make_label_tree(params):
        flat = traverse_util.flatten_dict(params, sep="/")

        def label_for_path(path_str):
            if "critic" in path_str:
                return "vf"
            elif "actor" in path_str:
                return "pi"
            elif "frozen" in path_str:
                return "frozen"
            return ""

        labels_flat = {k: label_for_path(k) for k in flat.keys()}
        return traverse_util.unflatten_dict(
            {tuple(k.split("/")): v for k, v in labels_flat.items()}
        )

    labels = make_label_tree(network_params)

    if config.gradient_clipping:
        tx = optax.partition(
            {
                "pi": optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.add_decayed_weights(config.l2_reg_pi),
                    optax.adam(config.alpha_pi, b1=config.adam_b1_pi, b2=config.adam_b2_pi, eps=config.adam_eps_pi)
                ),
                "vf": optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.add_decayed_weights(config.l2_reg_vf),
                    optax.adam(config.alpha_vf, b1=config.adam_b1_vf, b2=config.adam_b2_vf, eps=config.adam_eps_vf)
                ),
                "frozen": optax.set_to_zero(),
            },
            labels,
        )
    else:
        tx = optax.partition(
            {
                "pi": optax.chain(
                    optax.add_decayed_weights(config.l2_reg_pi),
                    optax.adam(config.alpha_pi, b1=config.adam_b1_pi, b2=config.adam_b2_pi, eps=config.adam_eps_pi)
                ),
                "vf": optax.chain(
                    optax.add_decayed_weights(config.l2_reg_vf),
                    optax.adam(config.alpha_vf, b1=config.adam_b1_vf, b2=config.adam_b2_vf, eps=config.adam_eps_vf)
                ),
                "frozen": optax.set_to_zero(),
            },
            labels,
        )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # L2-to-init: only allocate the frozen copy and multiplier tree when enabled.
    # When disabled, these are None and loss_fn's guard skips the regularisation.
    if config.use_l2_init:
        initial_params = jax.tree_util.tree_map(lambda p: p.copy(), network_params)

        def _make_l2_init_multipliers(params, labels, lambda_pi, lambda_vf):
            flat_params = traverse_util.flatten_dict(params, sep="/")
            flat_labels = traverse_util.flatten_dict(labels, sep="/")
            lam_map = {"pi": lambda_pi, "vf": lambda_vf}
            flat_mult = {
                k: jnp.array(lam_map.get(flat_labels[k], 0.0))
                for k in flat_params
            }
            return traverse_util.unflatten_dict(
                {tuple(k.split("/")): v for k, v in flat_mult.items()}
            )

        l2_init_multipliers = _make_l2_init_multipliers(
            network_params, labels, config.lambda_l2_init_pi, config.lambda_l2_init_vf,
        )
    else:
        initial_params = None
        l2_init_multipliers = None

    # Spectral regularisation: build per-leaf multiplier tree.
    # Every parameter is regularised (weights, biases, norm scales) — the
    # specific regularisation form for each leaf is decided at trace time in
    # loss_fn based on the parameter name / ndim.  When disabled, the
    # multiplier tree is None and the compile-time guard in loss_fn skips
    # the whole block.
    if config.use_spectral_reg:
        def _make_spectral_reg_multipliers(params, labels, lambda_pi, lambda_vf):
            flat_params = traverse_util.flatten_dict(params, sep="/")
            flat_labels = traverse_util.flatten_dict(labels, sep="/")
            lam_map = {"pi": lambda_pi, "vf": lambda_vf}
            flat_mult = {
                k: jnp.array(lam_map.get(flat_labels[k], 0.0))
                for k in flat_params
            }
            return traverse_util.unflatten_dict(
                {tuple(k.split("/")): v for k, v in flat_mult.items()}
            )

        spectral_reg_multipliers = _make_spectral_reg_multipliers(
            network_params, labels,
            config.lambda_spectral_pi, config.lambda_spectral_vf,
        )
    else:
        spectral_reg_multipliers = None

    ### Experiment
    def _zero_loss_info(config: TrainConfig):
        zeros = jnp.zeros((config.epochs, config.num_mini_batch), dtype=jnp.float32)
        return (zeros, (zeros, zeros, zeros))

    # Last-layer reset: reinitialize actor_mean and critic_value params + optimizer states.
    # The network is re-initialized from scratch with a fresh key; only the last-layer
    # leaves are swapped in (params and optimizer state).  Guarded by `config.use_reset`.
    if config.use_reset:
        # Pre-compute which leaves belong to the last layer, keyed by flattened path.
        _flat_params = traverse_util.flatten_dict(network_params, sep="/")
        _last_layer_keys = frozenset(
            k for k in _flat_params
            if "actor_mean" in k or "critic_value" in k
        )

        def _is_last_layer(path_str):
            return path_str in _last_layer_keys

        _last_layer_mask = traverse_util.unflatten_dict(
            {tuple(k.split("/")): _is_last_layer(k) for k in _flat_params}
        )

        def _reset_last_layer(train_state, rng):
            """Reinitialize last-layer params and their optimizer states."""
            rng, init_rng = jax.random.split(rng)
            # Get fresh random parameters for the entire network
            fresh_params = network.init(init_rng, init_hstate, init_x)

            # Selectively replace only last-layer params
            new_params = jax.tree_util.tree_map(
                lambda mask, fresh, old: jnp.where(mask, fresh, old),
                _last_layer_mask,
                fresh_params,
                train_state.params,
            )

            # Reinitialize optimizer state: build fresh opt state from new params,
            # then selectively swap only last-layer entries.
            fresh_opt_state = tx.init(new_params)
            new_opt_state = jax.tree_util.tree_map_with_path(
                lambda path, fresh_val, old_val: (
                    fresh_val if any(
                        (hasattr(k, "key") and (
                            "actor_mean" in k.key or "critic_value" in k.key
                        ))
                        for k in path
                    ) else old_val
                ),
                fresh_opt_state,
                train_state.opt_state,
            )

            train_state = train_state.replace(
                params=new_params,
                opt_state=new_opt_state,
            )
            return train_state, rng

    # Shrink and Perturb: shrink all params towards zero and add Gaussian noise,
    # then reinitialize optimizer state.  Guarded by `config.use_shrink_and_perturb`.
    if config.use_shrink_and_perturb:
        def _shrink_and_perturb(train_state, rng):
            """Apply shrink-and-perturb to all network parameters."""
            rng, noise_rng = jax.random.split(rng)

            def sp(path, p):
                leaf_rng = jax.random.fold_in(noise_rng, _keypath_crc32(path))
                noise = jax.random.normal(leaf_rng, shape=p.shape, dtype=p.dtype)
                return p * config.shrink_factor + noise * config.noise_scale

            new_params = jax.tree_util.tree_map_with_path(sp, train_state.params)

            # Reinitialize optimizer state for the perturbed parameters
            new_opt_state = tx.init(new_params)

            train_state = train_state.replace(
                params=new_params,
                opt_state=new_opt_state,
            )
            return train_state, rng

    env_step_state = (
        train_state,
        gymnax_state,
        log_env_state,
        config,
        obs,
        0,
        0,
        0,
        jnp.zeros((hint_dim,)),  # hint_trace
        rng,
        init_hstate,
    )

    @scan_tqdm(config.num_updates, print_rate=min(100, config.num_updates//20))
    def experiment_step(carry, iteration_idx):
        env_step_state, train_state, rng, initial_params, l2_init_multipliers, spectral_reg_multipliers = carry
        (
            train_state,
            gymnax_state,
            log_env_state,
            config,
            last_obs,
            last_action,
            last_reward,
            reward_trace,
            hint_trace,
            rng,
            hstate,
        ) = env_step_state
        start_timestep = log_env_state.timestep

        updates_per_video = (
            config.video_length + config.rollout_steps - 1
        ) // config.rollout_steps
        start_recording_update = config.num_updates - updates_per_video
        to_render = (iteration_idx >= start_recording_update) & config.allocate_frames

        gymnax_state = GymnaxEnvState.create(
            to_render=to_render,
            cond_render=gymnax_state.cond_render,
            env_step=gymnax_state.env_step,
            env_params=gymnax_state.env_params,
            env_state=gymnax_state.env_state,
        )

        env_step_state = (
            train_state,
            gymnax_state,
            log_env_state,
            config,
            last_obs,
            last_action,
            last_reward,
            reward_trace,
            hint_trace,
            rng,
            hstate,
        )

        # Roll out for config.rollout_steps
        env_step_state, traj_hstate_batch = jax.lax.scan(
            env_step, env_step_state, length=config.rollout_steps
        )
        traj_batch, hstate_batch = traj_hstate_batch
        (
            train_state,
            gymnax_state,
            log_env_state,
            config,
            last_obs,
            last_action,
            last_reward,
            reward_trace,
            hint_trace,
            rng,
            last_hstate,
        ) = env_step_state

        # Build last observation with previous action encoding
        action_encoded = jnp.zeros((4,))
        action_encoded = action_encoded.at[last_action].set(1)
        last_reward_encoded = jnp.expand_dims(last_reward, 0)

        if isinstance(last_obs, Mapping):
            obs_img = last_obs["image"]
            hint = last_obs["hint"]
            if config.use_hint_trace:
                hint_trace = (
                    config.reward_trace_decay * hint_trace
                    + (1.0 - config.reward_trace_decay) * hint
                )
                hint = hint_trace
            last_reward_encoded = jnp.concatenate((last_reward_encoded, hint), axis=-1)
        else:
            obs_img = last_obs

        sine = jnp.expand_dims(jnp.sin(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
        cosine = jnp.expand_dims(
            jnp.cos(2 * jnp.pi * log_env_state.timestep / PERIOD), 0
        )
        reward_trace = (
            config.reward_trace_decay * reward_trace
            + (1.0 - config.reward_trace_decay) * last_reward
        )
        reward_trace_encoded = jnp.expand_dims(reward_trace, 0)
        last_obs_encoded = (
            obs_img,
            action_encoded,
            last_reward_encoded,
            sine,
            cosine,
            reward_trace_encoded,
        )

        # Bootstrap value at last state
        _, _, last_value, _ = agent_step(
            last_obs_encoded, train_state, rng, last_hstate
        )
        last_val = last_value.squeeze()

        # Calculate GAE
        advantages, targets = calculate_gae(
            traj_batch, last_val, config.gamma, config.gae_lambda
        )

        # Conditionally perform the update based on how many env steps have elapsed.
        # If freeze_steps <= 0, updates are always performed.
        # Otherwise, once log_env_state.timestep exceeds freeze_steps, we stop updating.
        rng, update_rng = jax.random.split(rng)

        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            hstate_batch,
            advantages,
            targets,
            update_rng,
            config,
            initial_params,
            l2_init_multipliers,
            spectral_reg_multipliers,
        )

        def skip_update(update_state):
            (
                train_state,
                init_hstate,
                traj_batch,
                hstate_batch,
                advantages,
                targets,
                rng,
                config,
                _initial_params,
                _l2_init_multipliers,
                _spectral_reg_multipliers,
            ) = update_state
            return (train_state, rng), _zero_loss_info(config)

        should_update = jnp.logical_or(
            config.freeze_steps <= 0,
            log_env_state.timestep <= config.freeze_steps,
        )
        (train_state, rng), loss_info = jax.lax.cond(
            should_update, update_step, skip_update, update_state
        )

        # Periodically reset last layer (actor_mean + critic_value) params and optimizer
        # states.  Guarded at trace time by config.use_reset so no overhead when disabled.
        if config.use_reset:
            should_reset = should_update & _crossed_interval(
                start_timestep, log_env_state.timestep, config.reset_interval
            )
            train_state, rng = jax.lax.cond(
                should_reset,
                _reset_last_layer,
                lambda ts, r: (ts, r),
                train_state,
                rng,
            )

        # Periodically shrink all params and add noise, then reinit optimizer.
        # Guarded at trace time by config.use_shrink_and_perturb.
        if config.use_shrink_and_perturb:
            should_sp = should_update & _crossed_interval(
                start_timestep, log_env_state.timestep, config.sp_interval
            )
            train_state, rng = jax.lax.cond(
                should_sp,
                _shrink_and_perturb,
                lambda ts, r: (ts, r),
                train_state,
                rng,
            )

        # Collect a scalar reward summary for this iteration (mean reward over rollout)
        rewards = traj_batch.reward
        pos = traj_batch.info["pos"]
        biome_id = traj_batch.info["biome_id"]
        object_collected_id = traj_batch.info["object_collected_id"]
        biome_regret = traj_batch.info["biome_regret"]
        biome_rank = traj_batch.info["biome_rank"]
        if config.allocate_frames:
            # Update the frames buffer
            def update_frames(frames):
                idx_in_video = iteration_idx - start_recording_update
                start_idx = idx_in_video * config.rollout_steps
                return jax.lax.dynamic_update_slice(
                    frames, traj_batch.info["frame"], (start_idx, 0, 0, 0)
                )

            frames = jax.lax.cond(
                to_render, update_frames, lambda x: x, log_env_state.frames
            )
        else:
            frames = log_env_state.frames
        log_env_state = LogEnvState(
            returned_returns=log_env_state.returned_returns,
            timestep=log_env_state.timestep,
            frames=frames,
        )

        # Rebuild env_step_state for next iteration
        env_step_state = (
            train_state,
            gymnax_state,
            log_env_state,
            config,
            last_obs,
            last_action,
            last_reward,
            reward_trace,
            hint_trace,
            rng,
            last_hstate,
        )

        # Optional lightweight debug
        carry_out = (env_step_state, train_state, rng, initial_params, l2_init_multipliers, spectral_reg_multipliers)
        return carry_out, (
            rewards,
            pos,
            loss_info,
            biome_id,
            object_collected_id,
            biome_regret,
            biome_rank,
        )

    # Run training loop with lax.scan (collect per-iteration rewards)
    init_carry = (env_step_state, train_state, rng, initial_params, l2_init_multipliers, spectral_reg_multipliers)
    last_carry, info = jax.lax.scan(
        experiment_step,
        PBar(id=config.id, carry=init_carry),
        xs=jnp.arange(int(config.num_updates)),
    )
    rewards, pos, loss_info, biome_id, object_collected_id, biome_regret, biome_rank = (
        info
    )
    rewards = rewards.reshape((-1))
    pos = pos.reshape((-1, pos.shape[-1]))
    total_loss = jnp.mean(loss_info[0], axis=(-1, -2))
    value_loss = jnp.mean(loss_info[1][0], axis=(-1, -2))
    policy_loss = jnp.mean(loss_info[1][1], axis=(-1, -2))
    entropy = jnp.mean(loss_info[1][2], axis=(-1, -2))
    biome_id = biome_id.reshape((-1))
    object_collected_id = object_collected_id.reshape((-1))
    biome_regret = biome_regret.reshape((-1))
    biome_rank = biome_rank.reshape((-1))
    env_step_state = last_carry.carry[0]
    frames = env_step_state[2].frames
    return (
        rewards,
        pos,
        (total_loss, (value_loss, policy_loss, entropy)),
        biome_id,
        object_collected_id,
        biome_regret,
        biome_rank,
        frames,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", type=str, required=True)
    parser.add_argument("-i", "--idxs", nargs="+", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
    parser.add_argument("--silent", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=None)

    args = parser.parse_args()

    if not args.gpu:
        jax.config.update("jax_platform_name", "cpu")

    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logger = logging.getLogger("exp")
    prod = "cdr" in socket.gethostname() or args.silent
    if not prod:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # ----------------------
    # -- Experiment Def'n --
    # ----------------------
    timeout_handler = TimeoutHandler()

    exp = ExperimentModel.load(args.exp)

    try:
        indices = parse_indices(args.idxs, exp.numPermutations())
    except ValueError as e:
        parser.error(str(e))
    allocate_frames = len(indices) == 1

    # --------------------
    # -- Batch Set-up --
    # --------------------
    start_time = time.time()

    collectors = []
    rngs = []
    chks = []
    configs = []
    for idx in indices:
        chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
        chk.load_if_exists()
        timeout_handler.before_cancel(chk.save)
        chks.append(chk)

        collector = chk.build(
            "collector",
            lambda: Collector(
                # specify which keys to actually store and ultimately save
                # Options are:
                #  - Identity() (save everything)
                #  - Window(n)  take a window average of size n
                #  - Subsample(n) save one of every n elements
                config={
                    "ewm_reward": Pipe(
                        MovingAverage(0.999),
                        Subsample(max(exp.total_steps // 1000, 1)),
                    ),
                    "mean_ewm_reward": Last(
                        MovingAverage(0.999),
                        Mean(),
                    ),
                },
                # by default, ignore keys that are not explicitly listed above
                default=Ignore(),
            ),
        )
        collector.set_experiment_id(idx)
        collectors.append(collector)

        hypers = exp.get_hypers(idx)

        seed = exp.getRun(idx) + hypers.get("seed_offset", 0)
        rng = jax.random.PRNGKey(seed)

        freeze_steps = hypers.get("freeze_after_steps", hypers.get("freeze_steps", -1))
        # if "freeze_steps_end" in hypers:
        #     # TODO: this is to match the key for the env reset, that is the jax.random.PRNGKey(seed)[1] is used to reset the env
        #     freeze_key = jax.random.PRNGKey(seed + 42)
        #     freeze_steps_end = hypers["freeze_steps_end"]
        #     freeze_steps = jax.random.randint(freeze_key, (), freeze_steps, freeze_steps_end + 1)
        #     print(f"Freeze steps sampled to {freeze_steps}")

        rngs.append(rng)

        # derive num_updates if not explicitly present
        num_updates = (
            int(hypers["num_updates"])
            if "num_updates" in hypers
            else (exp.total_steps // int(hypers["rollout_steps"]) + 1)
        )
        if args.max_steps is not None:
            num_updates = args.max_steps
        reset_interval = hypers.get("reset_interval", hypers.get("reset_steps", -1))
        sp_interval = hypers.get("sp_interval", hypers.get("sp_steps", -1))
        config = TrainConfig(
            d_hidden=int(hypers["representation"]["d_hidden"]),
            agent_type=exp.agent,
            hidden_size=int(hypers["representation"]["hidden"]),
            rollout_steps=int(hypers["rollout_steps"]),
            epochs=int(hypers["epochs"]),
            num_mini_batch=int(hypers["num_mini_batch"]),
            gradient_clipping=bool(hypers["gradient_clipping"]),
            max_grad_norm=float(hypers["max_grad_norm"]),
            alpha_pi=float(hypers["optimizer_actor"]["alpha"]),
            alpha_vf=float(
                hypers["optimizer_critic"].get(
                    "alpha",
                    hypers["optimizer_critic"].get("lr_scale", jnp.nan)
                    * hypers["optimizer_actor"]["alpha"],
                )
            ),
            adam_eps_pi=float(hypers["optimizer_actor"]["eps"]),
            adam_eps_vf=float(hypers["optimizer_critic"]["eps"]),
            adam_b1_pi=float(hypers["optimizer_actor"].get("beta1", 0.9)),
            adam_b2_pi=float(hypers["optimizer_actor"].get("beta2", 0.999)),
            adam_b1_vf=float(hypers["optimizer_critic"].get("beta1", 0.9)),
            adam_b2_vf=float(hypers["optimizer_critic"].get("beta2", 0.999)),
            l2_reg_pi=float(hypers.get("l2_reg_pi", hypers.get("l2_reg", 0.0))),
            l2_reg_vf=float(hypers.get("l2_reg_vf", hypers.get("l2_reg", 0.0))),
            lambda_l2_init_pi=float(
                hypers.get("lambda_l2_init_pi", hypers.get("lambda_l2_init", 0.0))
            ),
            lambda_l2_init_vf=float(
                hypers.get("lambda_l2_init_vf", hypers.get("lambda_l2_init", 0.0))
            ),
            use_l2_init=bool(
                hypers.get("lambda_l2_init_pi", hypers.get("lambda_l2_init", 0.0))
                != 0.0
                or hypers.get("lambda_l2_init_vf", hypers.get("lambda_l2_init", 0.0))
                != 0.0
            ),
            lambda_spectral_pi=float(
                hypers.get("lambda_spectral_pi", hypers.get("lambda_spectral", 0.0))
            ),
            lambda_spectral_vf=float(
                hypers.get("lambda_spectral_vf", hypers.get("lambda_spectral", 0.0))
            ),
            use_spectral_reg=bool(
                hypers.get("lambda_spectral_pi", hypers.get("lambda_spectral", 0.0))
                != 0.0
                or hypers.get("lambda_spectral_vf", hypers.get("lambda_spectral", 0.0))
                != 0.0
            ),
            sparsity=hypers["representation"].get("sparsity", None),
            spectral_radius=hypers["representation"].get("spectral_radius", None),
            use_sinusoidal_encoding=bool(hypers.get("use_sinusoidal_encoding", False)),
            use_reward_trace=bool(hypers.get("use_reward_trace", hypers.get("representation", {}).get("use_reward_trace", False))),
            use_hint_trace=bool("_HT" in exp.agent),
            use_layernorm=bool(
                hypers.get(
                    "use_layernorm",
                    hypers.get("representation", {}).get("use_layernorm", False),
                )
            ),
            conv=str(hypers.get("representation", {}).get("conv", "Conv2D")),
            reward_trace_decay=float(hypers.get("reward_trace_decay", 1.0)),
            num_updates=num_updates,
            aperture_size=int(hypers["environment"]["aperture_size"]),
            render_mode=hypers["environment"].get("render_mode", "world_reward"),
            env_kwargs=tuple(
                sorted(
                    (k, v)
                    for k, v in hypers["environment"].items()
                    if k not in ["aperture_size", "env_id", "render_mode"]
                    and v is not None
                )
            ),
            env_id=hypers["environment"]["env_id"],
            gamma=float(hypers["gamma"]),
            gae_lambda=float(hypers["gae_lambda"]),
            clip_eps=float(hypers["clip_eps"]),
            vf_coef=float(hypers["vf_coef"]),
            entropy_coef=float(hypers["entropy_coef"]),
            id=idx,
            freeze_steps=int(freeze_steps),
            allocate_frames=allocate_frames,
            video_length=int(hypers.get("experiment", {}).get("video_length", 1000)),
            use_reset=bool(reset_interval > 0),
            reset_interval=int(reset_interval),
            use_shrink_and_perturb=bool(sp_interval > 0),
            sp_interval=int(sp_interval),
            shrink_factor=float(hypers.get("shrink_factor", 1.0)),
            noise_scale=float(hypers.get("noise_scale", 0.0)),
        )
        configs.append(config)

    batch_experiment = jax.vmap(experiment, in_axes=(0, 0))
    rngs = jnp.stack(rngs)
    configs_stacked = tree_map(lambda *xs: jnp.stack(xs), *configs)
    results = batch_experiment(rngs, configs_stacked)
    (
        rewards,
        pos,
        (total_loss, (value_loss, policy_loss, entropy)),
        biome_id,
        object_collected_id,
        biome_regret,
        biome_rank,
        frames,
    ) = results

    # --------------------
    # -- Saving --
    # --------------------
    total_collect_time = 0
    total_numpy_time = 0
    total_db_time = 0
    num_indices = len(indices)
    for i, idx in enumerate(indices):
        collector = collectors[i]
        chk = chks[i]
        config = configs[i]
        # process rewards for this run
        run_rewards = rewards[i]
        run_pos = pos[i]
        run_total_loss = total_loss[i]
        run_value_loss = value_loss[i]
        run_policy_loss = policy_loss[i]
        run_entropy = entropy[i]
        run_biome_id = biome_id[i]
        run_object_collected_id = object_collected_id[i]
        run_biome_regret = biome_regret[i]
        run_biome_rank = biome_rank[i]
        run_frames = frames[i]
        start_time = time.time()
        # for reward in run_rewards:
        #     collector.next_frame()
        #     collector.collect("ewm_reward", reward.item())
        #     collector.collect("mean_ewm_reward", reward.item())
        logger.debug(f"Mean rewards {run_rewards.mean()}")
        collector.reset()
        total_collect_time += time.time() - start_time

        # ------------
        # -- Saving --
        # ------------
        context = exp.buildSaveContext(idx, base=args.save_path)
        save_path = context.resolve("results.db")
        data_path = context.resolve(f"data/{idx}.npz")
        video_path = context.resolve(f"videos/{idx}")
        context.ensureExists(data_path, is_file=True)
        context.ensureExists(video_path, is_file=True)

        start_time = time.time()
        if config.allocate_frames:
            start_frame = (
                config.num_updates * config.rollout_steps - run_frames.shape[0]
            )
            end_frame = config.num_updates * config.rollout_steps
            save_video(
                list(run_frames),
                video_path,
                name_prefix=f"{start_frame}_{end_frame}",
                fps=8,
            )
        np.savez_compressed(
            data_path,
            rewards=run_rewards,
            pos=run_pos,
            total_loss=run_total_loss,
            value_loss=run_value_loss,
            policy_loss=run_policy_loss,
            entropy=run_entropy,
            biome_id=run_biome_id,
            object_collected_id=run_object_collected_id,
            biome_regret=run_biome_regret,
            biome_rank=run_biome_rank,
        )
        total_numpy_time += time.time() - start_time

        meta = getParamsAsDict(exp, idx)
        meta |= {"seed": exp.getRun(idx)}
        attach_metadata(save_path, idx, meta)

        start_time = time.time()
        collector.merge(context.resolve("results.db"))
        total_db_time += time.time() - start_time

        collector.close()
        chk.delete()
    logger.debug("--- Saving Timings ---")
    logger.debug(
        f"Total collect time: {total_collect_time:.4f}s | Average: {total_collect_time / num_indices:.4f}s"
    )
    logger.debug(
        f"Total numpy save time: {total_numpy_time:.4f}s | Average: {total_numpy_time / num_indices:.4f}s"
    )
    logger.debug(
        f"Total db save time: {total_db_time:.4f}s | Average: {total_db_time / num_indices:.4f}s"
    )
    total_save_time = total_collect_time + total_numpy_time + total_db_time
    logger.debug(
        f"Total save time: {total_save_time:.4f}s | Average: {total_save_time / num_indices:.4f}s"
    )


if __name__ == "__main__":
    main()
