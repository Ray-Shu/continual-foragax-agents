import os
import sys

sys.path.append(os.getcwd())
sys.path.insert(0, os.path.abspath("/tmp/src"))
import argparse
import gc
import logging
import lzma
import pickle
import socket
import time
import traceback
from collections.abc import Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.utils.save_video import save_video
from jax.tree_util import tree_map
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Ignore
from PyExpUtils.results.tools import getParamsAsDict
from tqdm.auto import tqdm

from environments.Foragax import Foragax
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.preempt import TimeoutHandler
from utils.rlglue import RlGlue


UNROLL = 1

# ------------------
# -- Command Args --
# ------------------
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


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True)
parser.add_argument("-i", "--idxs", nargs="+", type=str, required=True)
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
parser.add_argument("--silent", action="store_true", default=False)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=None)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
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

Problem = getProblem(exp.problem)

# --------------------
# -- Batch Set-up --
# --------------------
start_time = time.time()

collectors = []
glues = []
chks = []
first_hypers = None
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
            config={},
            # by default, ignore keys that are not explicitly listed above
            default=Ignore(),
        ),
    )
    collector.set_experiment_id(idx)
    collectors.append(collector)

    run = exp.getRun(idx)

    # set random seeds accordingly
    hypers = exp.get_hypers(idx)
    if not first_hypers:
        first_hypers = hypers

    # validate that shape changing hypers are static.
    assert hypers.get("batch") == first_hypers.get("batch")
    assert hypers.get("buffer_size") == first_hypers.get("buffer_size")
    assert hypers.get("buffer_min_size") == first_hypers.get("buffer_min_size")
    assert hypers.get("environment", {}).get("aperture_size") == first_hypers.get(
        "environment", {}
    ).get("aperture_size")
    assert hypers.get("n_step") == first_hypers.get("n_step")
    assert hypers.get("optimizer", {}).get("name") == first_hypers.get(
        "optimizer", {}
    ).get("name")
    assert hypers.get("representation", {}).get("type") == first_hypers.get(
        "representation", {}
    ).get("type")
    assert hypers.get("representation", {}).get("hidden") == first_hypers.get(
        "representation", {}
    ).get("hidden")

    # build stateful things and attach to checkpoint
    problem = chk.build("p", lambda: Problem(exp, idx, collector))
    agent = chk.build("a", problem.getAgent)
    env = chk.build("e", problem.getEnvironment)

    glue = chk.build("glue", lambda: RlGlue(agent, env))
    glues.append(glue)

assert first_hypers is not None

render_mode = first_hypers.get("experiment", {}).get("render_mode", "world_reward")


def render(carry):
    return glues[0].environment.env.render(
        carry.env_state.state, None, render_mode=render_mode
    )


if hasattr(glues[0].agent, "_step_without_update"):
    v_start = jax.vmap(glues[0]._start)
    v_step = jax.vmap(glues[0]._step)
    v_step_skip = jax.vmap(glues[0]._step_without_update)
    v_update_agent = jax.vmap(glues[0].agent._update_state_with_metrics_if_can_sample)
    v_advance_agent = jax.vmap(glues[0].agent._advance_update_clock)

    @jax.jit
    def v_step_no_update(state):
        state, interaction = v_step_skip(state)
        agent_state = v_advance_agent(state.agent_state)
        state = replace(state, agent_state=agent_state)
        return state, interaction

    @jax.jit
    def v_step_update(state):
        state, interaction = v_step_skip(state)
        agent_state = v_update_agent(state.agent_state)
        state = replace(state, agent_state=agent_state)
        agent_state = v_advance_agent(state.agent_state)
        state = replace(state, agent_state=agent_state)
        return state, interaction

    v_render = jax.vmap(render)
    use_explicit_update_steps = True
elif len(glues) > 1:
    # vmap glue methods
    v_start = jax.vmap(glues[0]._start)
    v_step = jax.vmap(glues[0]._step)
    v_render = jax.vmap(render)
    use_explicit_update_steps = False
else:
    v_start = glues[0]._start
    v_step = glues[0]._step
    v_render = render
    use_explicit_update_steps = False

total_setup_time = time.time() - start_time
num_indices = len(indices)
logger.debug("--- Batch Set-up Timings ---")
logger.debug(
    f"Total setup time: {total_setup_time:.4f}s | Average: {total_setup_time / num_indices:.4f}s"
)

n = exp.total_steps
if args.max_steps is not None:
    n = args.max_steps

video_length = first_hypers.get("experiment", {}).get("video_length", 0)
video_idxs = first_hypers.get("experiment", {}).get("video_idxs", [0])
if not any(idx in video_idxs for idx in indices) and num_indices > 1:
    video_length = 0

# --------------------
# -- Batch Execution --
# --------------------

start_step = None
save_every = first_hypers.get("experiment", {}).get("save_every", 10_001_000)
video_every = first_hypers.get("experiment", {}).get("video_every", save_every)
DORMANCY_KEYS = ("dormancy_pre_core", "dormancy_core", "dormancy_post_core")

datas = {}
datas["rewards"] = np.empty((len(indices), n), dtype=np.float16)
datas["weight_change"] = np.empty((len(indices), n), dtype=np.float16)
datas["squared_td_error"] = np.empty((len(indices), n), dtype=np.float16)
datas["abs_td_error"] = np.empty((len(indices), n), dtype=np.float16)
datas["loss"] = np.empty((len(indices), n), dtype=np.float16)

agent_metrics_obj = getattr(glues[0].agent.state, "metrics", None)
HAS_DORMANCY = agent_metrics_obj is not None and all(
    hasattr(agent_metrics_obj, k) for k in DORMANCY_KEYS
)
if HAS_DORMANCY:
    for k in DORMANCY_KEYS:
        datas[k] = np.empty((len(indices), n), dtype=np.float16)


def get_agent_metrics(agent_state, batch_shape):
    """Safely extract metrics from agent state, handling different agent types."""
    weight_change = jnp.zeros(batch_shape)
    squared_td_error = jnp.zeros(batch_shape)
    abs_td_error = jnp.zeros(batch_shape)
    loss = jnp.zeros(batch_shape)
    dormancy = {k: jnp.zeros(batch_shape) for k in DORMANCY_KEYS}

    if hasattr(agent_state, "metrics"):
        metrics = agent_state.metrics
        if hasattr(metrics, "weight_change"):
            weight_change = metrics.weight_change
        if hasattr(metrics, "squared_td_error"):
            squared_td_error = metrics.squared_td_error
        if hasattr(metrics, "abs_td_error"):
            abs_td_error = metrics.abs_td_error
        if hasattr(metrics, "loss"):
            loss = metrics.loss
        for k in DORMANCY_KEYS:
            if hasattr(metrics, k):
                dormancy[k] = getattr(metrics, k)

    return weight_change, squared_td_error, abs_td_error, loss, dormancy


if isinstance(glues[0].environment, Foragax):
    datas["pos"] = np.empty((len(indices), n, 2), dtype=np.int32)
    datas["biome_id"] = np.empty((len(indices), n), dtype=np.int32)
    datas["biome_regret"] = np.empty((len(indices), n), dtype=np.float16)
    datas["biome_rank"] = np.empty((len(indices), n), dtype=np.int32)
    datas["object_collected_id"] = np.empty((len(indices), n), dtype=np.int32)
    datas["hint"] = np.full((len(indices), n), -1, dtype=np.int32)

    if "Weather" in glues[0].environment.env.name:
        datas["temperatures"] = np.empty(
            (len(indices), n, len(glues[0].environment.env.objects)), dtype=np.float16
        )

    def get_data(carry, interaction):
        weight_change, squared_td_error, abs_td_error, loss, dormancy = (
            get_agent_metrics(carry.agent_state, interaction.reward.shape)
        )
        data = {
            "rewards": interaction.reward,
            "pos": carry.env_state.state.pos,
            "biome_id": interaction.extra["biome_id"],
            "biome_regret": interaction.extra["biome_regret"],
            "biome_rank": interaction.extra["biome_rank"],
            "object_collected_id": interaction.extra["object_collected_id"],
            "weight_change": weight_change,
            "squared_td_error": squared_td_error,
            "abs_td_error": abs_td_error,
            "loss": loss,
        }
        if HAS_DORMANCY:
            for k in DORMANCY_KEYS:
                data[k] = dormancy[k]

        if isinstance(interaction.obs, Mapping) and "hint" in interaction.obs:
            hint = interaction.obs["hint"]
            # hint is a one hot vector, store -1 if it is all zero.
            # store index of 1 if it is one-hot.
            hint_val = jnp.where(
                jnp.any(hint > 0, axis=-1), jnp.argmax(hint, axis=-1), -1
            )
            data["hint"] = hint_val
        elif "hint" in datas:
            data["hint"] = jnp.full(interaction.reward.shape, -1, dtype=jnp.int32)

        if "Weather" in glues[0].environment.env.name:
            data["temperatures"] = interaction.extra["temperatures"]
        return data
else:

    def get_data(carry, interaction):
        weight_change, squared_td_error, abs_td_error, loss, dormancy = (
            get_agent_metrics(carry.agent_state, interaction.reward.shape)
        )
        data = {
            "rewards": interaction.reward,
            "weight_change": weight_change,
            "squared_td_error": squared_td_error,
            "abs_td_error": abs_td_error,
            "loss": loss,
        }
        if HAS_DORMANCY:
            for k in DORMANCY_KEYS:
                data[k] = dormancy[k]
        return data


initial_glue_states = [g.state for g in glues]
loaded_checkpoints = [False] * len(indices)
checkpoint_load_failed = False


def reset_datas():
    fresh_datas = {}
    fresh_datas["rewards"] = np.empty((len(indices), n), dtype=np.float16)
    fresh_datas["weight_change"] = np.empty((len(indices), n), dtype=np.float16)
    fresh_datas["squared_td_error"] = np.empty((len(indices), n), dtype=np.float16)
    fresh_datas["abs_td_error"] = np.empty((len(indices), n), dtype=np.float16)
    fresh_datas["loss"] = np.empty((len(indices), n), dtype=np.float16)
    if HAS_DORMANCY:
        for k in DORMANCY_KEYS:
            fresh_datas[k] = np.empty((len(indices), n), dtype=np.float16)
    if isinstance(glues[0].environment, Foragax):
        fresh_datas["pos"] = np.empty((len(indices), n, 2), dtype=np.int32)
        fresh_datas["biome_id"] = np.empty((len(indices), n), dtype=np.int32)
        fresh_datas["biome_regret"] = np.empty((len(indices), n), dtype=np.float16)
        fresh_datas["biome_rank"] = np.empty((len(indices), n), dtype=np.int32)
        fresh_datas["object_collected_id"] = np.empty(
            (len(indices), n), dtype=np.int32
        )
        fresh_datas["hint"] = np.full((len(indices), n), -1, dtype=np.int32)
        if "Weather" in glues[0].environment.env.name:
            fresh_datas["temperatures"] = np.empty(
                (len(indices), n, len(glues[0].environment.env.objects)),
                dtype=np.float16,
            )
    return fresh_datas


for i, idx in enumerate(indices):
    context = exp.buildSaveContext(idx, base=args.checkpoint_path)
    step_path = context.resolve(f"{idx}/step.txt")
    glue_state_path = context.resolve(f"{idx}/glue_state.pkl.xz")
    data_path = context.resolve(f"{idx}/data.npz")
    if os.path.exists(step_path):
        with open(step_path, "r") as f:
            try:
                start_step_idx = int(f.read())
                if start_step is None:
                    start_step = start_step_idx
                else:
                    assert start_step == start_step_idx

                with lzma.open(glue_state_path, "rb") as f:
                    glues[i].state = pickle.load(f)

                with np.load(data_path) as data_idx:
                    for key in data_idx.keys():
                        datas[key][i, : len(data_idx[key])] = data_idx[key]
                loaded_checkpoints[i] = True
            except Exception:
                traceback.print_exc()
                logger.warning(
                    f"Failed to load checkpoint for index {idx}, starting fresh"
                )
                start_step = None
                checkpoint_load_failed = True
                datas = reset_datas()

if checkpoint_load_failed or (any(loaded_checkpoints) and not all(loaded_checkpoints)):
    missing = [idx for idx, loaded in zip(indices, loaded_checkpoints) if not loaded]
    logger.warning(
        "Not all selected indices had valid checkpoints; starting the whole batch fresh. "
        f"Missing checkpoint indices: {missing}"
    )
    start_step = None
    for glue, initial_state in zip(glues, initial_glue_states):
        glue.state = initial_state
    datas = reset_datas()

if len(glues) > 1:
    # combine states
    glue_states = tree_map(lambda *leaves: jnp.stack(leaves), *[g.state for g in glues])
elif use_explicit_update_steps:
    glue_states = tree_map(lambda leaf: jnp.expand_dims(leaf, 0), glues[0].state)
else:
    glue_states = glues[0].state

if start_step is None:
    glue_states, _ = v_start(glue_states)
    start_step = 0
else:
    logger.debug(f"Loaded checkpoints, resuming from step {start_step}")

current_step = start_step
training_pbar = tqdm(
    total=n,
    initial=current_step,
    desc="Training",
    unit="step",
    disable=args.silent,
)


def update_training_progress(step_count):
    step_count = int(step_count)
    remaining = training_pbar.total - training_pbar.n
    if remaining > 0:
        training_pbar.update(min(step_count, remaining))


def scan_progress(num_iters: int, unit_scale: int = 1):
    if args.silent:
        return lambda scan_fn: scan_fn

    print_rate = max(min(num_iters // 100, 1000), 1)

    def wrap(scan_fn):
        def wrapped(carry, x):
            result = scan_fn(carry, x)
            iter_num = x[0] if isinstance(x, tuple) else x
            next_iter = iter_num + 1
            should_update = (next_iter % print_rate == 0) | (next_iter == num_iters)
            iter_delta = jnp.where(
                next_iter == num_iters,
                ((num_iters - 1) % print_rate) + 1,
                print_rate,
            )
            step_delta = iter_delta * unit_scale

            def do_update(_):
                jax.debug.callback(
                    update_training_progress,
                    step_delta,
                    ordered=True,
                )
                return None

            jax.lax.cond(should_update, do_update, lambda _: None, operand=None)
            return result

        return wrapped

    return wrap


while current_step < n:
    next_save = ((current_step // save_every) + 1) * save_every
    next_video = ((current_step // video_every) + 1) * video_every
    next_milestone = min(next_save, next_video, n)
    steps_in_iter = next_milestone - current_step

    if steps_in_iter <= 0:
        break

    # We only record video if this milestone reaches a video frequency
    is_video_milestone = next_milestone % video_every == 0
    if is_video_milestone and video_length > 0:
        no_video_steps_count = max(steps_in_iter - video_length, 0)
        video_steps_count = min(steps_in_iter, video_length)
    else:
        no_video_steps_count = steps_in_iter
        video_steps_count = 0

    def video_step(carry, _):
        frame = v_render(carry)
        carry, interaction = v_step(carry)
        data = get_data(carry, interaction)
        return carry, (data, frame)

    data_chunk = None
    if no_video_steps_count > 0:
        update_freq = int(first_hypers.get("update_freq", 1))

        def step(carry, _):
            carry, interaction = v_step(carry)
            data = get_data(carry, interaction)
            return carry, data

        def no_update_step(carry, _):
            carry, interaction = v_step_no_update(carry)
            data = get_data(carry, interaction)
            return carry, data

        agent_step_count = int(np.asarray(glue_states.agent_state.steps).reshape(-1)[0])
        freeze_steps = first_hypers.get("freeze_steps", np.inf)
        if freeze_steps is None:
            freeze_steps = np.inf
        freeze_steps = float(freeze_steps)
        can_use_explicit_steps = (
            use_explicit_update_steps
            and update_freq > 1
            and agent_step_count + no_video_steps_count <= freeze_steps
        )

        if can_use_explicit_steps:
            data_chunks = []
            steps_remaining = no_video_steps_count

            steps_mod = agent_step_count % update_freq
            prefix_count = min((update_freq - steps_mod) % update_freq, steps_remaining)
            if prefix_count > 0:
                prefix_step = scan_progress(prefix_count)(no_update_step)
                glue_states, prefix_data = jax.lax.scan(
                    prefix_step, glue_states, jnp.arange(prefix_count), unroll=UNROLL
                )
                data_chunks.append(prefix_data)
                steps_remaining -= prefix_count

            block_count = steps_remaining // update_freq
            if block_count > 0:

                def update_block(carry, _):
                    carry, interaction = v_step_update(carry)
                    update_data = get_data(carry, interaction)
                    carry, skip_data = jax.lax.scan(
                        no_update_step,
                        carry,
                        jnp.arange(update_freq - 1),
                        unroll=UNROLL,
                    )
                    block_data = tree_map(
                        lambda first, rest: jnp.concatenate(
                            [jnp.expand_dims(first, 0), rest], axis=0
                        ),
                        update_data,
                        skip_data,
                    )
                    return carry, block_data

                glue_states, block_data = jax.lax.scan(
                    scan_progress(block_count, unit_scale=update_freq)(update_block),
                    glue_states,
                    jnp.arange(block_count),
                )
                block_data = tree_map(
                    lambda x: x.reshape((block_count * update_freq, *x.shape[2:])),
                    block_data,
                )
                data_chunks.append(block_data)
                steps_remaining -= block_count * update_freq

            if steps_remaining > 0:
                glue_states, tail_data = jax.lax.scan(
                    scan_progress(steps_remaining)(no_update_step),
                    glue_states,
                    jnp.arange(steps_remaining),
                    unroll=UNROLL,
                )
                data_chunks.append(tail_data)

            data_chunk = (
                data_chunks[0]
                if len(data_chunks) == 1
                else tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *data_chunks)
            )
        else:
            no_video_steps = jnp.arange(no_video_steps_count)
            glue_states, data_chunk = jax.lax.scan(
                scan_progress(no_video_steps_count)(step),
                glue_states,
                no_video_steps,
                unroll=UNROLL,
            )

    frames = None
    data_chunk_video = None
    if video_steps_count > 0:
        video_steps = jnp.arange(video_steps_count)
        glue_states, (data_chunk_video, frames) = jax.lax.scan(
            scan_progress(video_steps_count)(video_step),
            glue_states,
            video_steps,
            unroll=UNROLL,
        )
        if data_chunk is None:
            data_chunk = data_chunk_video
        else:
            data_chunk = tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0),
                data_chunk,
                data_chunk_video,
            )
        frames = np.asarray(frames)
        if len(glues) < 2 and use_explicit_update_steps:
            frames = frames[:, 0]

    # checkpointing
    if len(glues) < 2 and not use_explicit_update_steps:
        data_chunk = tree_map(lambda x: np.expand_dims(x, 1), data_chunk)

    for i, idx in enumerate(indices):
        data_idx = tree_map(lambda x: x[:, i], data_chunk)
        for key in datas:
            datas[key][i, current_step : current_step + steps_in_iter] = data_idx[key]

        if next_milestone % save_every == 0 and next_milestone < n:
            checkpoint_start_time = time.time()
            context = exp.buildSaveContext(idx, base=args.checkpoint_path)

            # Write to temporary files first for atomic checkpointing
            glue_state_path_tmp = context.resolve(f"{idx}/glue_state.pkl.xz.tmp")
            data_path_tmp = context.resolve(f"{idx}/data.tmp.npz")
            step_path_tmp = context.resolve(f"{idx}/step.txt.tmp")

            # Ensure directories exist
            context.ensureExists(glue_state_path_tmp, is_file=True)
            context.ensureExists(data_path_tmp, is_file=True)
            context.ensureExists(step_path_tmp, is_file=True)

            # Prepare checkpoint data
            if len(glues) > 1 or use_explicit_update_steps:
                glue_state_idx = tree_map(lambda x: x[i], glue_states)
            else:
                glue_state_idx = glue_states

            data_to_save = tree_map(lambda d: d[i, :next_milestone], datas)

            # Write temporary files
            with lzma.open(glue_state_path_tmp, "wb") as f:
                pickle.dump(glue_state_idx, f)

            np.savez_compressed(data_path_tmp, **data_to_save)

            with open(step_path_tmp, "w") as f:
                f.write(str(next_milestone))

            # Atomically rename temporary files to final names
            final_glue_state_path = context.resolve(f"{idx}/glue_state.pkl.xz")
            final_data_path = context.resolve(f"{idx}/data.npz")
            final_step_path = context.resolve(f"{idx}/step.txt")

            os.rename(glue_state_path_tmp, final_glue_state_path)
            os.rename(data_path_tmp, final_data_path)
            os.rename(step_path_tmp, final_step_path)
            checkpoint_time = time.time() - checkpoint_start_time
            logger.debug(
                f"Checkpointed {idx} at {next_milestone} in {checkpoint_time:.4f}s"
            )

        # Save video
        if (
            (next_milestone % video_every == 0 or next_milestone == n)
            and video_length
            and frames is not None
        ):
            if idx in video_idxs:
                video_context = exp.buildSaveContext(idx, base=args.save_path)
                video_path = video_context.resolve(f"videos/{idx}")
                video_context.ensureExists(video_path, is_file=True)

                frames_list = list(frames)
                logger.debug(
                    f"Saving {next_milestone - len(frames_list)}_{next_milestone} video to {video_path}"
                )
                save_video(
                    frames_list,
                    video_path,
                    name_prefix=f"{next_milestone - len(frames_list)}_{next_milestone}",
                    fps=8,
                )

    current_step = next_milestone
    if "frames" in locals():
        del frames
    if "data_chunk" in locals():
        del data_chunk
    if "data_chunk_video" in locals():
        del data_chunk_video
    gc.collect()

training_pbar.close()

# --------------------
# -- Saving --
# --------------------
total_numpy_time = 0
total_db_time = 0
num_indices = len(indices)
for i, idx in enumerate(indices):
    collector = collectors[i]

    # ------------
    # -- Saving --
    # ------------
    context = exp.buildSaveContext(idx, base=args.save_path)
    save_path = context.resolve("results.db")
    data_path = context.resolve(f"data/{idx}.npz")
    context.ensureExists(data_path, is_file=True)

    start_time = time.time()
    data_to_save = tree_map(lambda d: d[i], datas)
    np.savez_compressed(data_path, **data_to_save)
    total_numpy_time += time.time() - start_time

    meta = getParamsAsDict(exp, idx)
    meta |= {"seed": exp.getRun(idx)}
    attach_metadata(save_path, idx, meta)

    start_time = time.time()
    collector.merge(context.resolve("results.db"))
    total_db_time += time.time() - start_time

    collector.close()

logger.debug("--- Saving Timings ---")
logger.debug(
    f"Total numpy save time: {total_numpy_time:.4f}s | Average: {total_numpy_time / num_indices:.4f}s"
)
logger.debug(
    f"Total db save time: {total_db_time:.4f}s | Average: {total_db_time / num_indices:.4f}s"
)
total_save_time = total_numpy_time + total_db_time
logger.debug(
    f"Total save time: {total_save_time:.4f}s | Average: {total_save_time / num_indices:.4f}s"
)
