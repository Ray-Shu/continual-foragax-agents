"""DQN trained with UPGD instead of Adam.

UPGD (Utility-based Perturbed Gradient Descent) replaces Adam wholesale; it
needs the raw gradient as F_l and a per-weight diagonal Hessian approximation
as S_l. We get S_l from HesScale, which runs as a separate custom backward pass
(see `hesscale`) alongside the usual `jax.grad` step.

Current limitations
-------------------
- Network must be a strict feedforward chain ``x -> q-vector``. HesScale's
  recursion does not yet model the scalar-feature input path, so this agent
  asserts ``representation.scalar_features=""`` in your config.
- ``arch_spec`` is hardcoded for the network types in ``_build_arch_spec``.
  At ``__init__`` time we print ``self.state.params`` keys so you can verify
  the Haiku module paths match what HesScale will walk; if they don't, edit
  ``_build_arch_spec``.
- UPGD is used standalone here (NOT chained with Adam/SWR). Any ``optimizer``
  / ``swr`` config keys are ignored when ``upgd`` is set.
"""

from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from ml_instrumentation.Collector import Collector

from algorithms.nn.DQN import DQN
from hesscale import SCALARS_JOIN, hesscale, lin_dense, make_conv_lin
from optimizers import upgd_optimizer
from utils.jax import huber_loss


# -----------------------------------------------------------------------------
# Hypers
# -----------------------------------------------------------------------------
@dataclass
class UPGDHypers:
    step_size: float            # alpha
    utility_decay_rate: float   # beta
    noise: float                # sigma  (std of the Gaussian perturbation)
    seed: int


# -----------------------------------------------------------------------------
# Per-sample loss used only to seed HesScale (g_L, s_L) via autodiff at the
# network output. The actual training loss for `jax.grad` still goes through
# DQN._loss; we keep the two consistent by using the same Huber form here.
# -----------------------------------------------------------------------------
def q_loss_for_hesscale(q_out: jax.Array, action: jax.Array, td_target: jax.Array):
    """Huber loss as a function of the Q-vector only.

    `td_target` is r + gamma * max(Q_target(s')); the caller stop_gradients it.
    """
    return huber_loss(1.0, q_out[action], td_target)


# -----------------------------------------------------------------------------
# Architecture-spec lookup: maps the network name in the config to the ordered
# list of (path, lin_fn, activation_hint) entries (interspersed with SCALARS_JOIN
# where the network concatenates scalar features partway through).
#
# Paths are TUPLES of dict keys (not slash-strings) because Haiku stores some
# inner modules under flat keys that themselves contain '/' or '~'
# (e.g. ForagerNet -> "phi/~/phi"). state.params is structured as
#     {"phi": <feat_params>, "<head_name>": <head_params>}
# so the path for the feature-net's first conv is ("phi", "phi/~/phi"), etc.
# -----------------------------------------------------------------------------
def _build_arch_spec(network_name: str, scalars_size: int, layers: int = None):
    if network_name == "TwoLayerRelu":
        return [
            (("phi", "phi"),   lin_dense, "relu"),
            (("phi", "phi_1"), lin_dense, "relu"),
            (("q",   "q"),     lin_dense, "linear"),
        ]
    if network_name == "OneLayerRelu":
        return [
            (("phi", "phi"), lin_dense, "relu"),
            (("q",   "q"),   lin_dense, "linear"),
        ]
    if network_name == "ForagerNet":
        # Specifically for layers=N, default conv (Conv2D 3x3 stride 1 SAME),
        # no LayerNorm, no pre/post-core MLPs. ForagerNet only concatenates
        # scalars when scalars_size > 0 (matching its own forward).
        if layers is None or layers < 1:
            raise NotImplementedError(
                f"DQN_UPGD: ForagerNet arch_spec requires 'representation.layers' "
                f"to be a positive int (got {layers!r})."
            )
        spec = [
            (("phi", "phi/~/phi"),
             make_conv_lin(stride=1, padding="SAME"), "relu"),
        ]
        if scalars_size > 0:
            spec.append(SCALARS_JOIN)
        # The N core_mlp Linears: first is "phi/~/linear", then "phi/~/linear_1", ...
        for i in range(layers):
            name = "phi/~/linear" if i == 0 else f"phi/~/linear_{i}"
            spec.append((("phi", name), lin_dense, "relu"))
        spec.append((("q", "q"), lin_dense, "linear"))
        return spec
    raise NotImplementedError(
        f"DQN_UPGD: no arch_spec for network type '{network_name}'. "
        "Add it to _build_arch_spec in DQN_UPGD.py "
        "(inspect self.state.params to see the actual module names)."
    )


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
class DQN_UPGD(DQN):
    """DQN whose optimizer is UPGD, with HesScale providing S_l."""

    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        # Parse UPGD hypers BEFORE super().__init__: NNAgent.__init__ calls
        # self._build_optimizer during construction, and our override reads
        # these attributes.
        upgd_params = params.get("upgd")
        if upgd_params is None:
            raise ValueError(
                "DQN_UPGD requires an 'upgd' dict in params with keys "
                "step_size, utility_decay_rate, noise."
            )
        self._upgd_hypers = UPGDHypers(
            step_size=float(upgd_params["step_size"]),
            utility_decay_rate=float(upgd_params["utility_decay_rate"]),
            noise=float(upgd_params["noise"]),
            seed=int(upgd_params.get("seed", seed)),
        )

        super().__init__(observations, actions, params, collector, seed)

        # Build the arch_spec from the network name + scalars_size + layers.
        # Networks that concatenate scalar features partway (ForagerNet) get a
        # SCALARS_JOIN entry; pure feedforward nets (TwoLayerRelu) do not.
        network_name = params["representation"]["type"]
        network_layers = params["representation"].get("layers")
        self.arch_spec = _build_arch_spec(
            network_name, self.scalars_size, network_layers
        )

        # Sanity print so the user can confirm Haiku's actual module paths
        # line up with arch_spec. Fires once at agent init (not under JIT).
        print(f"[DQN_UPGD] network={network_name}  scalars_size={self.scalars_size}")
        print(f"[DQN_UPGD] state.params structure:")
        for k, v in self.state.params.items():
            if isinstance(v, dict):
                for kk in v:
                    print(f"           ({k!r}, {kk!r}): w={v[kk]['w'].shape} b={v[kk]['b'].shape}")
            else:
                print(f"           {k}: <leaf>")
        print("[DQN_UPGD] arch_spec entries:")
        for entry in self.arch_spec:
            print(f"           {entry}")

    # -------------------------------------------------------------------------
    # Optimizer: replace Adam(+SWR) with UPGD.
    # -------------------------------------------------------------------------
    def _build_optimizer(self, optimizer_hypers, swr_hypers):
        # optimizer_hypers / swr_hypers are intentionally ignored; UPGD is
        # standalone (it consumes the raw gradient as F_l, so chaining with
        # Adam would corrupt the utility signal).
        return upgd_optimizer(
            step_size=self._upgd_hypers.step_size,
            utility_decay_rate=self._upgd_hypers.utility_decay_rate,
            noise=self._upgd_hypers.noise,
            seed=self._upgd_hypers.seed,
        )

    # -------------------------------------------------------------------------
    # Bootstrap TD target per sample, stop_gradient'd. Pulled out of _loss so
    # HesScale can reuse it without re-running through _loss.
    # -------------------------------------------------------------------------
    def _compute_td_targets(self, target_params: hk.Params, batch: Dict):
        xp = batch["x"][:, -1]
        scalars_p = batch["scalars"][:, -1]

        rs = batch["r"]
        gs = batch["gamma"]
        gs = jnp.concatenate([jnp.ones((gs.shape[0], 1)), gs[:, :-1]], axis=1)
        gs = jnp.cumprod(gs, axis=1)
        r = jnp.sum(rs[:, :-1] * gs[:, :-1], axis=1)
        g = gs[:, -1]

        phi_p = self.phi(target_params, xp, scalars=scalars_p).out
        qsp = self.q(target_params, phi_p)
        vp = qsp.max(axis=-1)
        return jax.lax.stop_gradient(r + g * vp)

    # -------------------------------------------------------------------------
    # Update step: same shape as DQN._computeUpdate, with HesScale folded in.
    # -------------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state, batch: Dict):
        # First-order gradient: unchanged from DQN (Huber TD-error loss).
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch)

        # HesScale: diagonal Hessian approximation alongside.
        x = batch["x"][:, 0]
        actions = batch["a"][:, 0]
        td_targets = self._compute_td_targets(state.target_params, batch)
        scalars = batch["scalars"][:, 0] if self.scalars_size > 0 else None
        _, hess_diag = hesscale(
            self.arch_spec,
            state.params,
            x,
            q_loss_for_hesscale,
            actions,
            td_targets,
            scalars_batch=scalars,
        )

        # UPGD consumes grad + hess_diag via extra_args.
        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)
        updates, new_optim = optimizer.update(
            grad, state.optim, state.params,
            grad=grad, hess_diag=hess_diag,
        )
        new_params = optax.apply_updates(state.params, updates)

        flat_updates, _ = ravel_pytree(updates)
        weight_change = jnp.linalg.norm(flat_updates, ord=1)
        metrics["weight_change"] = weight_change

        return replace(state, params=new_params, optim=new_optim), metrics
