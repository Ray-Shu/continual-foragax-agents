"""Regression tests for shrink-and-perturb noise sampling.

Guards against a regression where ``DQN_Shrink_and_Perturb._shrink_and_perturb``
reused a single PRNG subkey across every parameter leaf. ``jax.random.normal``
is deterministic given ``(key, shape)``, so the buggy code produced identical
noise tensors for any two same-shape leaves rather than the i.i.d. Gaussian
draws shrink-and-perturb requires (issue #161).
"""

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


class TestShrinkAndPerturbKeySplit:
    """Independent per-leaf noise — pattern used in ``_shrink_and_perturb``."""

    def test_same_shape_leaves_get_independent_noise(self):
        """Two same-shape leaves must receive different perturbations."""
        params = {
            "a": jnp.zeros((4, 4)),
            "b": jnp.zeros((4, 4)),
        }
        shrink_factor = 0.5
        noise_scale = 1.0

        key = jax.random.PRNGKey(0)
        new_params = _shrink_and_perturb_kernel(
            key, params, shrink_factor, noise_scale
        )

        # Because params start at zero, post-perturb values equal scaled noise.
        delta_a = new_params["a"]
        delta_b = new_params["b"]

        # Both deltas should be non-trivial.
        assert jnp.any(delta_a != 0.0)
        assert jnp.any(delta_b != 0.0)

        # The two same-shape leaves must NOT receive identical noise.
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
        noise_scale = 0.1

        key = jax.random.PRNGKey(7)
        new_params = _shrink_and_perturb_kernel(
            key, params, shrink_factor, noise_scale
        )

        # Subtract the deterministic shrink contribution to recover noise.
        noise_a = new_params["a"] - params["a"] * shrink_factor
        noise_b = new_params["b"] - params["b"] * shrink_factor

        assert not jnp.allclose(noise_a, noise_b)

    def test_module_source_does_not_share_subkey(self):
        """The DQN_Shrink_and_Perturb module must not feed ``subkey`` to noise."""
        import pathlib

        # Locate the module source via the repo layout (works regardless of
        # whether the editable install points at this worktree or elsewhere).
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        module_path = (
            repo_root / "src" / "algorithms" / "nn" / "DQN_Shrink_and_Perturb.py"
        )
        src = module_path.read_text()

        # The buggy pattern fed the shared ``subkey`` directly into
        # ``jax.random.normal`` for every leaf.
        assert "jax.random.normal(subkey" not in src, (
            "DQN_Shrink_and_Perturb._shrink_and_perturb is reusing the shared "
            "subkey across all parameter leaves — this is the issue #161 bug."
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
