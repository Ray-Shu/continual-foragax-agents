"""Verification suite for the Step-8 refactor:
   A) TwoLayerRelu still verifies (sanity: wrt-h carry == wrt-a carry for no-join nets)
   B) scalars-join works on a tiny ForagerNet-shaped Haiku net
   C) DQN_UPGD instantiates + updates with ForagerNet + scalars (Tier 0)
"""
import sys; sys.path.insert(0, "src")

import jax
import jax.numpy as jnp
import haiku as hk

from hesscale import (
    SCALARS_JOIN, hesscale, lin_dense, make_conv_lin,
)


# =============================================================================
# Test A -- TwoLayerRelu (no join). Sanity-check the wrt-h carry refactor.
# =============================================================================
print("\n=== Test A: TwoLayerRelu MLP, grads match jax.grad =========")

def mlp_fn(x):
    x = hk.Linear(8, name="h0")(x); x = jax.nn.relu(x)
    x = hk.Linear(6, name="h1")(x); x = jax.nn.relu(x)
    x = hk.Linear(3, name="out")(x)
    return x

mlp = hk.without_apply_rng(hk.transform(mlp_fn))
hk_params_A = mlp.init(jax.random.PRNGKey(0), jnp.zeros(4))
print("  param keys:", list(hk_params_A.keys()))    # ['h0', 'h1', 'out']

ARCH_A = [
    (("h0",),  lin_dense, "relu"),
    (("h1",),  lin_dense, "relu"),
    (("out",), lin_dense, "linear"),
]

B = 5
kx, ka, kt = jax.random.split(jax.random.PRNGKey(1), 3)
X_A       = jax.random.normal(kx, (B, 4))
actions_A = jax.random.randint(ka, (B,), 0, 3)
targets_A = jax.random.normal(kt, (B,)) * 0.1

def dqn_loss(q_out, action, target):
    return 0.5 * (q_out[action] - target) ** 2

grads_hs_A, _ = hesscale(
    ARCH_A, hk_params_A, X_A, dqn_loss, actions_A, targets_A,
)

def batched_loss_A(params, X, a, t):
    q = jax.vmap(lambda x: mlp.apply(params, x))(X)
    return jnp.mean(jax.vmap(dqn_loss)(q, a, t))

auto_A = jax.grad(batched_loss_A)(hk_params_A, X_A, actions_A, targets_A)

all_ok_A = True
for k in ("h0", "h1", "out"):
    w_ok = bool(jnp.allclose(auto_A[k]["w"], grads_hs_A[k]["w"], atol=1e-5))
    b_ok = bool(jnp.allclose(auto_A[k]["b"], grads_hs_A[k]["b"], atol=1e-5))
    print(f"  {k}:  w={w_ok}   b={b_ok}")
    all_ok_A &= w_ok and b_ok
print(f"  Test A: {'PASS' if all_ok_A else 'FAIL'}")


# =============================================================================
# Test B -- ForagerNet-shaped: Conv -> ReLU -> Flatten -> concat(scalars)
#                              -> Linear -> ReLU -> Linear (linear out).
# This is the new code path -- forward join + backward split.
# =============================================================================
print("\n=== Test B: Conv + scalars_join + Linear, grads match jax.grad ====")

def conv_net_fn(x, scalars):
    # x: (B, H, W, C),  scalars: (B, S)  -- hk.Conv2D treats leading dim as batch
    h = hk.Conv2D(16, 3, 1, padding="SAME", name="conv")(x)
    h = jax.nn.relu(h)
    h = hk.Flatten(preserve_dims=1)(h)                  # (B, H*W*16)
    h = jnp.concatenate([h, scalars], axis=-1)           # (B, H*W*16 + S)
    h = hk.Linear(64, name="lin")(h)
    h = jax.nn.relu(h)
    h = hk.Linear(3, name="head")(h)
    return h

conv_net = hk.without_apply_rng(hk.transform(conv_net_fn))
# Batched dummies (batch=1) so hk.Conv2D infers the spatial dims correctly.
hk_params_B = conv_net.init(
    jax.random.PRNGKey(0), jnp.zeros((1, 4, 4, 1)), jnp.zeros((1, 5))
)
print("  param keys:", list(hk_params_B.keys()))    # ['conv', 'lin', 'head']

ARCH_B = [
    (("conv",), make_conv_lin(1, "SAME"), "relu"),
    SCALARS_JOIN,
    (("lin",),  lin_dense, "relu"),
    (("head",), lin_dense, "linear"),
]

B_b = 5
kx, ks, ka, kt = jax.random.split(jax.random.PRNGKey(2), 4)
X_B       = jax.random.normal(kx, (B_b, 4, 4, 1))
S_B       = jax.random.normal(ks, (B_b, 5))
actions_B = jax.random.randint(ka, (B_b,), 0, 3)
targets_B = jax.random.normal(kt, (B_b,)) * 0.1

grads_hs_B, hess_hs_B = hesscale(
    ARCH_B, hk_params_B, X_B, dqn_loss, actions_B, targets_B, scalars_batch=S_B,
)

def batched_loss_B(params, X, S, a, t):
    # conv_net is inherently batched (hk.Conv2D needs the batch dim), so call directly.
    q = conv_net.apply(params, X, S)            # (B, 3)
    return jnp.mean(jax.vmap(dqn_loss)(q, a, t))

auto_B = jax.grad(batched_loss_B)(hk_params_B, X_B, S_B, actions_B, targets_B)

all_ok_B = True
for k in ("conv", "lin", "head"):
    w_ok = bool(jnp.allclose(auto_B[k]["w"], grads_hs_B[k]["w"], atol=1e-4))
    b_ok = bool(jnp.allclose(auto_B[k]["b"], grads_hs_B[k]["b"], atol=1e-4))
    print(f"  {k}:  w={w_ok}   b={b_ok}")
    all_ok_B &= w_ok and b_ok

# Also check hess returned in the right shape and no NaNs (HesScale is approximate
# for conv; we don't allclose against jax.hessian here -- the recursion correctness
# is established by grads matching, plus we already verified the conv approximation
# in earlier throwaway steps).
hess_finite_B = all(
    bool(jnp.all(jnp.isfinite(hess_hs_B[k][p])))
    for k in ("conv", "lin", "head") for p in ("w", "b")
)
print(f"  hess_diag finite: {hess_finite_B}")
print(f"  Test B: {'PASS' if (all_ok_B and hess_finite_B) else 'FAIL'}")


# =============================================================================
# Test C -- DQN_UPGD on a real ForagerNet config (E138-shaped Tier 0).
# =============================================================================
print("\n=== Test C: DQN_UPGD instantiation + one update with ForagerNet ===")

from algorithms.nn.DQN_UPGD import DQN_UPGD
from ml_instrumentation.Collector import Collector

actions_C = 3
params_C = {
    "representation": {"type": "ForagerNet", "hidden": 64, "layers": 1},
    # scalar_features defaults to "hint,last_action,last_reward" -- scalars_size > 0
    "upgd":           {"step_size": 1e-3, "utility_decay_rate": 0.9, "noise": 1e-4},
    "target_refresh": 1000,
    "epsilon": 0.1,
    "optimizer": {"alpha": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "buffer_size": 1000, "batch": 32, "total_steps": 1000, "update_freq": 1,
    "n_step": 1,
}
# observations as a plain tuple -> no hint, scalars_size = actions + 1 = 4
agent = DQN_UPGD((7, 7, 3), actions_C, params_C, Collector(), seed=0)
print(f"  agent.scalars_size = {agent.scalars_size}")

# Fake one batch matching what DQN._loss expects.
B_c, T = params_C["batch"], 2
kx, ka, kr = jax.random.split(jax.random.PRNGKey(3), 3)
batch = {
    "x":       jax.random.normal(kx, (B_c, T, 7, 7, 3)),
    "scalars": jnp.zeros((B_c, T, agent.scalars_size)),
    "a":       jax.random.randint(ka, (B_c, T), 0, actions_C),
    "r":       jax.random.normal(kr, (B_c, T)) * 0.1,
    "gamma":   jnp.full((B_c, T), 0.99),
}

state_before = agent.state
state_after, metrics = agent._computeUpdate(state_before, batch)

loss_c          = float(metrics["loss"])
wc_c            = float(metrics["weight_change"])
changed_c       = not bool(jnp.allclose(
    state_before.params["phi"]["phi/~/phi"]["w"],
    state_after.params["phi"]["phi/~/phi"]["w"],
))
finite_c = bool(jnp.all(jnp.isfinite(state_after.params["q"]["q"]["w"])))

print(f"  loss          = {loss_c}")
print(f"  weight_change = {wc_c}")
print(f"  params changed: {changed_c}")
print(f"  no NaNs       : {finite_c}")
print(f"  Test C: {'PASS' if (changed_c and finite_c and jnp.isfinite(wc_c)) else 'FAIL'}")


# =============================================================================
# Summary
# =============================================================================
print("\n=== Summary ===")
print(f"  A (TwoLayerRelu):       {'PASS' if all_ok_A else 'FAIL'}")
print(f"  B (Conv + scalars_join): {'PASS' if (all_ok_B and hess_finite_B) else 'FAIL'}")
print(f"  C (DQN_UPGD ForagerNet): "
      f"{'PASS' if (changed_c and finite_c and jnp.isfinite(wc_c)) else 'FAIL'}")
