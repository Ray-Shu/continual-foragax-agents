"""Tests for the RTU eligibility-trace norm metric in ``utils.ppo_metrics``."""

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.nn.ACMLP import ActorCriticMLP
from algorithms.nn.RealTimeACMLP import RealTimeActorCriticMLP
from algorithms.nn.RealTimeACMLPMulti import RealTimeActorCriticMLPMulti
from utils.ppo_metrics import has_rtu_trace, nan_trace_norm, trace_norm

BATCH, D_HIDDEN, D_INPUT = 1, 8, 5


def _fill(tree, key):
    """Replace every zero leaf of a carry sub-tree with standard normal noise."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef,
        [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves, strict=True)],
    )


def _random_carry(seed=0):
    """A RealTimeActorCriticMLP carry with noise in both the recurrent state and
    the trace, so a metric that wrongly includes the state is detectable."""
    hstate = RealTimeActorCriticMLP.initialize_memory(BATCH, D_HIDDEN, D_INPUT)
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    (h_a, e_a), (h_c, e_c) = hstate
    return (
        (_fill(h_a, keys[0]), _fill(e_a, keys[1])),
        (_fill(h_c, keys[2]), _fill(e_c, keys[3])),
    )


def _reference(hstate):
    """Hand-computed (pi, vf, total) L2 norms over the trace leaves only."""

    def flat_l2(trace):
        return float(
            np.sqrt(sum(np.sum(np.square(np.asarray(x))) for x in trace))
        )

    pi, vf = flat_l2(hstate[0][1]), flat_l2(hstate[1][1])
    return pi, vf, float(np.sqrt(pi**2 + vf**2))


class TestHasRTUTrace:
    def test_rtu_agent_has_trace(self):
        hstate = RealTimeActorCriticMLP.initialize_memory(BATCH, D_HIDDEN, D_INPUT)
        assert has_rtu_trace(hstate)

    def test_multi_cell_rtu_agent_has_trace(self):
        hstate = RealTimeActorCriticMLPMulti.initialize_memory(
            BATCH, D_HIDDEN, D_INPUT
        )
        assert len(hstate) == 4  # (actor1, critic1, actor2, critic2)
        assert has_rtu_trace(hstate)

    def test_feedforward_agent_has_no_trace(self):
        # ActorCriticMLP.initialize_memory returns None -- nothing to measure.
        assert not has_rtu_trace(ActorCriticMLP.initialize_memory(BATCH, D_HIDDEN, D_INPUT))


class TestTraceNorm:
    def test_zero_at_init(self):
        hstate = RealTimeActorCriticMLP.initialize_memory(BATCH, D_HIDDEN, D_INPUT)
        assert np.allclose([float(v) for v in trace_norm(hstate)], 0.0)

    def test_matches_hand_computed_l2(self):
        hstate = _random_carry()
        assert np.allclose(
            [float(v) for v in trace_norm(hstate)], _reference(hstate), rtol=1e-5
        )

    def test_total_pools_both_trunks(self):
        pi, vf, total = (float(v) for v in trace_norm(_random_carry()))
        assert np.isclose(total, np.sqrt(pi**2 + vf**2), rtol=1e-5)

    def test_ignores_the_recurrent_state(self):
        """Only the trace is measured: perturbing h leaves the norm unchanged."""
        hstate = _random_carry()
        before = [float(v) for v in trace_norm(hstate)]
        perturbed = (
            (tuple(x + 100.0 for x in hstate[0][0]), hstate[0][1]),
            hstate[1],
        )
        assert np.allclose([float(v) for v in trace_norm(perturbed)], before, rtol=1e-6)

    def test_trunks_are_measured_separately(self):
        """Scaling only the critic trace moves vf and total, not pi."""
        hstate = _random_carry()
        pi, vf, _ = (float(v) for v in trace_norm(hstate))
        scaled = (hstate[0], (hstate[1][0], tuple(2.0 * x for x in hstate[1][1])))
        pi2, vf2, _ = (float(v) for v in trace_norm(scaled))
        assert np.isclose(pi2, pi, rtol=1e-6)
        assert np.isclose(vf2, 2.0 * vf, rtol=1e-5)

    def test_traceable_under_jit_and_cond(self):
        """The shape rtu_ppo calls it in: a lax.cond against the NaN branch."""
        hstate = _random_carry()

        @jax.jit
        def gated(h, flag):
            return jax.lax.cond(
                flag, lambda _: trace_norm(h), lambda _: nan_trace_norm(), None
            )

        assert np.allclose(
            [float(v) for v in gated(hstate, True)], _reference(hstate), rtol=1e-5
        )
        assert all(np.isnan(float(v)) for v in gated(hstate, False))

    def test_multi_cell_alternates_actor_critic(self):
        """For the 4-trunk carry, cells 0/2 are actor and 1/3 critic."""
        hstate = RealTimeActorCriticMLPMulti.initialize_memory(
            BATCH, D_HIDDEN, D_INPUT
        )
        ones = jax.tree_util.tree_map(jnp.ones_like, hstate)
        pi, vf, _ = (float(v) for v in trace_norm(ones))
        # Symmetric carry: equal per-cell trace sizes, so the two trunks match.
        assert np.isclose(pi, vf, rtol=1e-6)
        n_per_cell = 4 * D_HIDDEN + 4 * D_INPUT * D_HIDDEN
        assert np.isclose(pi, np.sqrt(2 * n_per_cell), rtol=1e-5)
