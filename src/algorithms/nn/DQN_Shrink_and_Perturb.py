from dataclasses import replace
from functools import partial
import zlib
from typing import Dict

import jax
import jax.lax
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers


def _keypath_crc32(path) -> int:
    path_str = "/".join(str(getattr(entry, "key", entry)) for entry in path)
    return zlib.crc32(path_str.encode("utf-8"))


@cxu.dataclass
class Hypers(BaseHypers):
    sp_steps: int
    shrink_factor: float
    noise_scale: float


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers


class DQN_Shrink_and_Perturb(DQN):
    def __init__(
        self,
        observations: tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        hypers = Hypers(
            **self.state.hypers.__dict__,
            sp_steps=params["sp_steps"],
            shrink_factor=params["shrink_factor"],
            noise_scale=params["noise_scale"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            hypers=hypers,
        )

    @partial(jax.jit, static_argnums=0)
    def _maybe_update(self, state: AgentState) -> AgentState:
        state = super()._maybe_update(state)
        state = jax.lax.cond(
            (state.steps % state.hypers.sp_steps == 0) & (state.steps > 0),
            self._shrink_and_perturb,
            lambda s: s,
            state,
        )
        return state

    @partial(jax.jit, static_argnums=0)
    def _shrink_and_perturb(self, state: AgentState):
        key, subkey = jax.random.split(state.key)

        def sp(path, p):
            leaf_key = jax.random.fold_in(subkey, _keypath_crc32(path))
            noise = jax.random.normal(leaf_key, shape=p.shape, dtype=p.dtype)
            return p * state.hypers.shrink_factor + noise * state.hypers.noise_scale

        params = jax.tree_util.tree_map_with_path(sp, state.params)

        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)
        optim = optimizer.init(params)

        return replace(state, key=key, params=params, optim=optim)
