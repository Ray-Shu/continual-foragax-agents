"""NTK and churn plasticity metrics for PPO agents.

Unlike the DQN path in ``continuing_main.py`` -- where a Python-level milestone
loop can pause training and call into numpy -- the PPO training loop in
``rtu_ppo.py`` runs entirely inside a single ``jax.lax.scan`` that is then
``jax.vmap``'d across runs.  Every metric therefore has to be a *pure JAX*
function with statically-shaped outputs so it can be traced inside the scan and
gated with ``jax.lax.cond``.

The PPO network ``apply_fn`` has the signature::

    hidden, pi, value = apply_fn(params, hidden, obs)

where ``obs`` is the 6-tuple
``(image, action_encoded, last_reward, sine, cosine, reward_trace)`` (each with
a leading batch axis), ``pi`` is a ``distrax.Categorical`` over ``action_dim``
actions, and ``value`` has shape ``(batch,)``.

Two heads are measured separately:

* **value (critic)** -- scalar value output, the direct analogue of the DQN
  Q-value metrics.
* **policy (actor)** -- the ``action_dim`` policy logits.

For each head we report the NTK Gram-matrix rank and condition number, plus the
per-update *churn*: the norm of the change in the head's predictions on a fixed
reference batch from immediately before to immediately after one PPO update.

Separately -- on its own cadence and config flag, *not* bundled with the NTK /
churn metrics above -- we report a single global **weight norm**: the L2 norm of
*all* network parameters (shared torso + both heads) after the update.  It needs
neither the reference batch nor a Jacobian, so it is essentially free and is
gated independently (see ``weight_norm`` / ``nan_weight_norm`` and the
``weight_norm_freq`` config in ``rtu_ppo.py``).  A growing weight norm is a
common plasticity-loss signature.
"""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def build_ref_obs_tuple(x: jnp.ndarray, action_dim: int, reward_dim: int) -> Tuple:
    """Build the 6-tuple obs the PPO network expects for a single reference obs.

    Scalar / context features (action encoding, last reward, sinusoidal time
    encoding, reward trace) are zeroed, mirroring the DQN reference-metric setup
    in ``utils.metrics.compute_ntk_metrics`` which feeds zero scalars.  A leading
    batch axis of size 1 is added so the convolutional / dense layers see the
    rank they expect.

    Args:
        x: A single reference observation image (no batch axis).
        action_dim: Number of discrete actions (size of the action encoding).
        reward_dim: Width of the ``last_reward`` feature (1 for plain envs,
            ``1 + hint_dim`` for hint envs).

    Returns:
        The 6-tuple ``(image, action_encoded, last_reward, sine, cosine,
        reward_trace)``, each with a leading batch axis of size 1.
    """
    image = jnp.expand_dims(x, 0)
    action_encoded = jnp.zeros((1, action_dim))
    last_reward = jnp.zeros((1, reward_dim))
    sine = jnp.zeros((1, 1))
    cosine = jnp.zeros((1, 1))
    reward_trace = jnp.zeros((1, 1))
    return (image, action_encoded, last_reward, sine, cosine, reward_trace)


def _flatten_jacobian(jac_tree: Any, n_rows: int) -> jnp.ndarray:
    """Flatten a Jacobian pytree to a dense ``[n_rows, n_params]`` matrix."""
    leaves = jax.tree_util.tree_leaves(jac_tree)
    flat_leaves = [leaf.reshape(n_rows, -1) for leaf in leaves]
    return jnp.concatenate(flat_leaves, axis=1)


def weight_norm(params: Any) -> jnp.ndarray:
    """Global L2 norm of every parameter leaf in ``params``.

    Concatenating-then-norming is unnecessary: ``sqrt(sum of per-leaf squared
    sums)`` equals the norm of the fully flattened parameter vector.  Computed on
    the post-update parameters, matching the NTK convention.

    Returns:
        A scalar array holding ``||theta||_2``.
    """
    leaves = jax.tree_util.tree_leaves(params)
    sq_sum = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    return jnp.sqrt(sq_sum)


def nan_weight_norm() -> jnp.ndarray:
    """The ``NaN`` weight-norm scalar emitted on non-metric / disabled updates."""
    return jnp.float32(jnp.nan)


def _ntk_rank_cond(jac_flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """NTK Gram-matrix rank and condition number from a flat Jacobian.

    The NTK Gram matrix is ``J Jᵀ`` (shape ``[n_rows, n_rows]``).  Parameter
    columns that do not influence the head (e.g. actor params for the value
    output) contribute zero rows to ``J`` and so leave ``J Jᵀ`` unchanged.
    """
    ntk = jac_flat @ jac_flat.T
    rank = jnp.linalg.matrix_rank(ntk).astype(jnp.float32)
    cond = jnp.linalg.cond(ntk)
    return rank, cond


def value_metrics(
    apply_fn: Callable,
    params_before: Any,
    params_after: Any,
    init_hstate: Any,
    x_ref: jnp.ndarray,
    action_dim: int,
    reward_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NTK rank / condition number and per-update churn for the value head.

    NTK is measured on the post-update parameters (the network's current state,
    matching the DQN convention).  Churn is the norm of the value change on
    ``x_ref`` from ``params_before`` to ``params_after``.

    Returns:
        ``(rank, cond, churn)`` as scalar arrays.
    """

    def value_of(params, x):
        obs = build_ref_obs_tuple(x, action_dim, reward_dim)
        _, _, value = apply_fn(params, init_hstate, obs)
        return value[0]

    # Per-sample predictions before / after the update -> churn.
    pred_before = jax.vmap(value_of, in_axes=(None, 0))(params_before, x_ref)
    pred_after = jax.vmap(value_of, in_axes=(None, 0))(params_after, x_ref)
    churn = jnp.linalg.norm(pred_after - pred_before)

    # NTK on the current (post-update) params.
    n_ref = x_ref.shape[0]
    jac = jax.vmap(jax.jacrev(value_of, argnums=0), in_axes=(None, 0))(
        params_after, x_ref
    )
    jac_flat = _flatten_jacobian(jac, n_ref)
    rank, cond = _ntk_rank_cond(jac_flat)
    return rank, cond, churn


def policy_metrics(
    apply_fn: Callable,
    params_before: Any,
    params_after: Any,
    init_hstate: Any,
    x_ref: jnp.ndarray,
    action_dim: int,
    reward_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NTK rank / condition number and per-update churn for the policy head.

    The policy output is the ``action_dim`` logit vector, so the Jacobian has
    ``n_ref * action_dim`` rows.  Churn is measured on the action probabilities
    (softmax of the logits), the interpretable notion of how much the policy
    moved on the reference set in one update.

    Returns:
        ``(rank, cond, churn)`` as scalar arrays.
    """

    def logits_of(params, x):
        obs = build_ref_obs_tuple(x, action_dim, reward_dim)
        _, pi, _ = apply_fn(params, init_hstate, obs)
        return pi.logits[0]  # (action_dim,)

    def probs_of(params, x):
        obs = build_ref_obs_tuple(x, action_dim, reward_dim)
        _, pi, _ = apply_fn(params, init_hstate, obs)
        return pi.probs[0]  # (action_dim,)

    pred_before = jax.vmap(probs_of, in_axes=(None, 0))(params_before, x_ref)
    pred_after = jax.vmap(probs_of, in_axes=(None, 0))(params_after, x_ref)
    churn = jnp.linalg.norm(pred_after - pred_before)

    n_ref = x_ref.shape[0]
    # jacrev of a vector output -> leaves carry a leading action_dim axis;
    # vmap over samples adds the n_ref axis in front: [n_ref, action_dim, ...].
    jac = jax.vmap(jax.jacrev(logits_of, argnums=0), in_axes=(None, 0))(
        params_after, x_ref
    )
    jac_flat = _flatten_jacobian(jac, n_ref * action_dim)
    rank, cond = _ntk_rank_cond(jac_flat)
    return rank, cond, churn


def compute_ppo_metrics(
    apply_fn: Callable,
    params_before: Any,
    params_after: Any,
    init_hstate: Any,
    x_ref: jnp.ndarray,
    action_dim: int,
    reward_dim: int,
    compute_value: bool = True,
    compute_policy: bool = True,
) -> Tuple[jnp.ndarray, ...]:
    """Compute value- and policy-head NTK + churn metrics.

    Pure JAX and statically shaped so it can be traced inside ``jax.lax.scan``
    / ``jax.lax.cond`` and vmapped across runs.  Heads that are disabled (or
    when this is called on a non-metric step via ``lax.cond``) report ``NaN``.

    The weight norm is intentionally *not* computed here -- it has its own
    cadence and config flag; see ``weight_norm`` / ``nan_weight_norm``.

    Args:
        apply_fn: The network ``apply`` function.
        params_before: Parameters before the current PPO update (for churn).
        params_after: Parameters after the current PPO update (NTK + churn).
        init_hstate: Initial hidden state sized for batch 1 (zeros for RTUs).
        x_ref: Reference observation images, shape ``[n_ref, ...]``.
        action_dim: Number of discrete actions.
        reward_dim: Width of the ``last_reward`` feature.
        compute_value: Whether to measure the value head.
        compute_policy: Whether to measure the policy head.

    Returns:
        ``(value_rank, value_cond, value_churn, policy_rank, policy_cond,
        policy_churn)`` as scalar arrays.
    """
    nan = jnp.float32(jnp.nan)

    if compute_value:
        v_rank, v_cond, v_churn = value_metrics(
            apply_fn, params_before, params_after, init_hstate,
            x_ref, action_dim, reward_dim,
        )
    else:
        v_rank, v_cond, v_churn = nan, nan, nan

    if compute_policy:
        p_rank, p_cond, p_churn = policy_metrics(
            apply_fn, params_before, params_after, init_hstate,
            x_ref, action_dim, reward_dim,
        )
    else:
        p_rank, p_cond, p_churn = nan, nan, nan

    return v_rank, v_cond, v_churn, p_rank, p_cond, p_churn


def nan_ppo_metrics() -> Tuple[jnp.ndarray, ...]:
    """The all-``NaN`` NTK / churn metric tuple emitted on non-metric updates."""
    nan = jnp.float32(jnp.nan)
    return nan, nan, nan, nan, nan, nan
