from dataclasses import replace
from functools import partial
from typing import Dict

import jax
import jax.lax
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers


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

        leaves, treedef = jax.tree_util.tree_flatten(state.params)
        leaf_keys = jax.random.split(subkey, len(leaves))
        keys_tree = jax.tree_util.tree_unflatten(treedef, leaf_keys)

        def sp(k, p):
            noise = jax.random.normal(k, shape=p.shape, dtype=p.dtype)
            return p * state.hypers.shrink_factor + noise * state.hypers.noise_scale

        params = jax.tree_util.tree_map(sp, keys_tree, state.params)

        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)
        optim = optimizer.init(params)

        return replace(state, key=key, params=params, optim=optim)
