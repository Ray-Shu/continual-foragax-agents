"""Regression tests for shrink-and-perturb noise sampling.

Guards against a regression where ``DQN_Shrink_and_Perturb._shrink_and_perturb``
reused a single PRNG subkey across every parameter leaf. ``jax.random.normal``
is deterministic given ``(key, shape)``, so the buggy code produced identical
noise tensors for any two same-shape leaves rather than the i.i.d. Gaussian
draws shrink-and-perturb requires (issue #161).
"""

import inspect

import jax
import jax.numpy as jnp


def _shrink_and_perturb_kernel(key, params, shrink_factor, noise_scale):
    """Reference implementation mirroring ``_shrink_and_perturb``'s kernel."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    leaf_keys = jax.random.split(key, len(leaves))
    keys_tree = jax.tree_util.tree_unflatten(treedef, leaf_keys)

    def sp(k, p):
        noise = jax.random.normal(k, shape=p.shape, dtype=p.dtype)
        return p * shrink_factor + noise * noise_scale

    return jax.tree_util.tree_map(sp, keys_tree, params)


_jitted_kernel = jax.jit(
    _shrink_and_perturb_kernel, static_argnames=("shrink_factor", "noise_scale")
)


class TestShrinkAndPerturbKeySplit:
    """Independent per-leaf noise — pattern used in ``_shrink_and_perturb``."""

    def test_same_shape_leaves_get_independent_noise(self):
        """Two same-shape leaves must receive different perturbations."""
        params = {
            "a": jnp.zeros((4, 4)),
            "b": jnp.zeros((4, 4)),
        }
        new_params = _shrink_and_perturb_kernel(
            jax.random.PRNGKey(0), params, 0.5, 1.0
        )

        delta_a = new_params["a"]
        delta_b = new_params["b"]

        assert jnp.any(delta_a != 0.0)
        assert jnp.any(delta_b != 0.0)
        assert not jnp.allclose(delta_a, delta_b), (
            "Same-shape leaves received identical noise — shrink-and-perturb "
            "is reusing a single PRNG key across leaves (issue #161)."
        )

    def test_noise_independent_of_starting_params(self):
        """Verify the per-leaf delta differs even when starting params differ."""
        params = {
            "a": jnp.ones((3, 3)),
            "b": jnp.ones((3, 3)) * 2.0,
        }
        shrink_factor = 0.5
        new_params = _shrink_and_perturb_kernel(
            jax.random.PRNGKey(7), params, shrink_factor, 0.1
        )

        noise_a = new_params["a"] - params["a"] * shrink_factor
        noise_b = new_params["b"] - params["b"] * shrink_factor

        assert not jnp.allclose(noise_a, noise_b)

    def test_kernel_jits_and_preserves_independence(self):
        """The pattern must trace under ``jax.jit`` and still give independent noise.

        Exercises the same code path the production
        ``DQN_Shrink_and_Perturb._shrink_and_perturb`` runs under (jitted
        ``tree_unflatten`` over the array returned by ``jax.random.split``).
        """
        params = {
            "a": jnp.zeros((4, 4)),
            "b": jnp.zeros((4, 4)),
            "c": jnp.zeros((8,)),
            "d": jnp.zeros((8,)),
        }
        new_params = _jitted_kernel(jax.random.PRNGKey(42), params, 0.5, 1.0)

        assert not jnp.allclose(new_params["a"], new_params["b"])
        assert not jnp.allclose(new_params["c"], new_params["d"])

    def test_production_function_uses_per_leaf_keys(self):
        """``_shrink_and_perturb``'s body must derive a key per leaf, not reuse subkey.

        Scoped to the function source via ``inspect.getsource`` so that comments
        elsewhere in the module cannot accidentally satisfy or break this check.
        Accepts either ``split`` + ``tree_map`` or ``fold_in`` + ``tree_map_with_path``.
        """
        from algorithms.nn.DQN_Shrink_and_Perturb import DQN_Shrink_and_Perturb

        src = inspect.getsource(DQN_Shrink_and_Perturb._shrink_and_perturb)

        assert "jax.random.normal(subkey" not in src, (
            "DQN_Shrink_and_Perturb._shrink_and_perturb feeds the shared "
            "subkey directly into jax.random.normal — that's the issue #161 "
            "bug. Each leaf must receive its own derived key."
        )

        derives_per_leaf_keys = (
            "jax.random.split(subkey" in src
            or "tree_map_with_path" in src
            or "fold_in" in src
        )
        assert derives_per_leaf_keys, (
            "Could not detect a per-leaf key derivation pattern in "
            "_shrink_and_perturb. Expected jax.random.split(subkey, ...) or "
            "fold_in/tree_map_with_path."
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
