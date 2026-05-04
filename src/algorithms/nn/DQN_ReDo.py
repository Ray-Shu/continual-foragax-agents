from dataclasses import replace
from functools import partial
from typing import Dict, Tuple

import jax
import jax.lax
import jax.numpy as jnp
from ml_instrumentation.Collector import Collector
import optax

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers

@cxu.dataclass
class Hypers(BaseHypers):
    redo_freq: int
    redo_threshold: float


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers  # type: ignore


def reset_momentum(momentum: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.where(mask, jnp.zeros_like(momentum), momentum)


class DQN_ReDo(DQN):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        rep_type = self.rep_params["type"]
        if rep_type != "ForagerNet":
            raise NotImplementedError(
                f"DQN_ReDo only supports ForagerNet (conv='None', layers=2). "
                f"Got {rep_type!r}."
            )
        if self.rep_params.get("conv", "Conv2D") != "None":
            raise NotImplementedError(
                f"DQN_ReDo on ForagerNet only supports conv='None'; "
                f"got conv={self.rep_params.get('conv')!r}."
            )
        if self.rep_params.get("layers", 0) != 2:
            raise NotImplementedError(
                f"DQN_ReDo on ForagerNet only supports layers=2; "
                f"got layers={self.rep_params.get('layers')}."
            )
        if params.get("swr") is not None:
            raise NotImplementedError("DQN_ReDo does not support combining with SWR.")

        self._reset_ln = bool(params.get("redo_reset_layernorm", True))
        self._use_ln = bool(self.rep_params.get("use_layernorm", False))

        hypers = Hypers(
            **self.state.hypers.__dict__,
            redo_freq=int(params["redo_freq"]),
            redo_threshold=float(params["redo_threshold"]),
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            hypers=hypers,
        )

    def _score(self, activation: jax.Array) -> jax.Array:
        reduce_axes = tuple(range(activation.ndim - 1))
        mean_activation = jnp.mean(jnp.abs(activation), axis=reduce_axes)
        score = mean_activation / (jnp.mean(mean_activation) + 1e-9)
        return score

    def _dormant(self, activation: jax.Array, threshold: float) -> jax.Array:
        return self._score(activation) <= threshold

    @partial(jax.jit, static_argnums=0)
    def _redo_step(self, state: AgentState) -> AgentState:
        key, sample_key, init_key = jax.random.split(state.key, 3)

        batch = self.buffer.sample(state.buffer_state, sample_key)
        x = batch.experience["x"][:, 0]
        scalars = batch.experience["scalars"][:, 0]

        feat = self.phi(state.params, x, scalars=scalars)
        threshold = state.hypers.redo_threshold
        dormant_d1 = self._dormant(feat.activations["relu"], threshold)
        dormant_d2 = self._dormant(feat.activations["relu_1"], threshold)

        # Mask trees mirror state.params; init to all-False.
        incoming_mask = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=bool), state.params
        )
        outgoing_mask = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=bool), state.params
        )

        phi = state.params["phi"]

        # ---- Dense_1 (first hidden Linear) ----
        incoming_mask["phi"]["phi/~/linear"]["w"] = jnp.broadcast_to(
            dormant_d1[None, :], phi["phi/~/linear"]["w"].shape
        )
        incoming_mask["phi"]["phi/~/linear"]["b"] = dormant_d1
        if self._use_ln and self._reset_ln:
            incoming_mask["phi"]["phi/~/layer_norm"]["scale"] = dormant_d1
            incoming_mask["phi"]["phi/~/layer_norm"]["offset"] = dormant_d1
        outgoing_mask["phi"]["phi/~/linear_1"]["w"] = jnp.broadcast_to(
            dormant_d1[:, None], phi["phi/~/linear_1"]["w"].shape
        )

        # ---- Dense_2 (last hidden Linear) ----
        incoming_mask["phi"]["phi/~/linear_1"]["w"] = jnp.broadcast_to(
            dormant_d2[None, :], phi["phi/~/linear_1"]["w"].shape
        )
        incoming_mask["phi"]["phi/~/linear_1"]["b"] = dormant_d2
        if self._use_ln and self._reset_ln:
            incoming_mask["phi"]["phi/~/layer_norm_1"]["scale"] = dormant_d2
            incoming_mask["phi"]["phi/~/layer_norm_1"]["offset"] = dormant_d2
        # Outgoing: rows of q/w
        outgoing_mask["q"]["q"]["w"] = jnp.broadcast_to(
            dormant_d2[:, None], state.params["q"]["q"]["w"].shape
        )

        leaves, treedef = jax.tree.flatten(state.params)
        keys = jax.random.split(init_key, len(leaves))
        keys_tree = jax.tree.unflatten(treedef, list(keys))

        def _sample_new_param(path, init_fn, param, key):
            leaf_name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
            if leaf_name == "w":
                return init_fn(key, param.shape, param.dtype)
            return jnp.zeros_like(param)

        new_params = jax.tree_util.tree_map_with_path(
            _sample_new_param, self.initializers, state.params, keys_tree
        )

        def apply_redo(path, param, new_param, in_mask, out_mask):
            leaf_name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
            if leaf_name == "w":
                param = jnp.where(in_mask, new_param, param)
            elif leaf_name == "b":
                param = jnp.where(in_mask, jnp.zeros_like(param), param)
            elif leaf_name == "scale":
                param = jnp.where(in_mask, jnp.ones_like(param), param)
            elif leaf_name == "offset":
                param = jnp.where(in_mask, jnp.zeros_like(param), param)
            param = jnp.where(out_mask, jnp.zeros_like(param), param)
            return param

        new_params = jax.tree_util.tree_map_with_path(
            apply_redo, state.params, new_params, incoming_mask, outgoing_mask
        )

        # reset mu, nu of adam optimizer
        mask = jax.tree.map(lambda i, o: i | o, incoming_mask, outgoing_mask)
        adam_state: optax.ScaleByAdamState = state.optim[0]  # type: ignore
        new_mu = jax.tree.map(reset_momentum, adam_state.mu, mask)
        new_nu = jax.tree.map(reset_momentum, adam_state.nu, mask)
        new_adam_state = optax.ScaleByAdamState(
            adam_state.count, mu=new_mu, nu=new_nu
        )
        new_optim = (new_adam_state, *state.optim[1:])  # type: ignore

        return replace(state, key=key, params=new_params, optim=new_optim)

    @partial(jax.jit, static_argnums=0)
    def _maybe_update(self, state: AgentState) -> AgentState:
        state = super()._maybe_update(state)
        do_redo = (
            (state.updates > 0)
            & (state.updates % state.hypers.redo_freq == 0)
            & self.buffer.can_sample(state.buffer_state)
        )
        state = jax.lax.cond(do_redo, self._redo_step, lambda s: s, state)
        return state
