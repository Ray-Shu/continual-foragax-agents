import jax.numpy as jnp
from flax import nnx

# Gating layers from GTrXL (Parisotto et al. 2019, "Stabilizing Transformers for RL").
# 
# The paper sweeps the bias (roughly 0.1 to 2) and finds it matters a lot.
#
# The "residual" gating is the default. 

GATES = ("residual", "input", "output", "highway", "sigtanh", "gru")

def _linear(d_hidden:int, rngs:nnx.Rngs):
    return nnx.Linear(in_features=d_hidden, out_features=d_hidden, use_bias=False, rngs=rngs)

class ResidualGate(nnx.Module):
    """g(x, y) = x + y -- the ungated baseline. Carries no parameters."""
    def __call__(self, x, y):
        return x + y

class InputGate(nnx.Module):
    """g(x, y) = sigmoid(W_g x) * x + y.

    The only gate in the paper with no b_g, so it has no identity-map init.
    """
    def __init__(self, d_hidden:int, rngs:nnx.Rngs):
        self.gate_linear = _linear(d_hidden, rngs)

    def __call__(self, x, y):
        return nnx.sigmoid(self.gate_linear(x)) * x + y

class OutputGate(nnx.Module):
    """g(x, y) = x + sigmoid(W_g x - b_g) * y."""
    def __init__(self, d_hidden:int, rngs:nnx.Rngs, bias_init:float):
        self.gate_linear = _linear(d_hidden, rngs)
        self.gate_bias = nnx.Param(jnp.full((d_hidden,), bias_init))

    def __call__(self, x, y):
        return x + nnx.sigmoid(self.gate_linear(x) - self.gate_bias) * y

class HighwayGate(nnx.Module):
    """g(x, y) = h * x + (1 - h) * y, where h = sigmoid(W_g x + b_g)."""
    def __init__(self, d_hidden:int, rngs:nnx.Rngs, bias_init:float):
        self.gate_linear = _linear(d_hidden, rngs)
        self.gate_bias = nnx.Param(jnp.full((d_hidden,), bias_init))

    def __call__(self, x, y):
        h = nnx.sigmoid(self.gate_linear(x) + self.gate_bias)
        return h * x + (1 - h) * y

class SigTanhGate(nnx.Module):
    """g(x, y) = x + sigmoid(W_g y - b_g) * tanh(U_g y)."""
    def __init__(self, d_hidden:int, rngs:nnx.Rngs, bias_init:float):
        self.gate_linear = _linear(d_hidden, rngs)
        self.candidate_linear = _linear(d_hidden, rngs)
        self.gate_bias = nnx.Param(jnp.full((d_hidden,), bias_init))

    def __call__(self, x, y):
        gate = nnx.sigmoid(self.gate_linear(y) - self.gate_bias)
        return x + gate * nnx.tanh(self.candidate_linear(y))

class GRUGate(nnx.Module):
    """GRU-type gating -- the "GTrXL" variant, best-performing in the paper.

        r = sigmoid(W_r y + U_r x)
        z = sigmoid(W_z y + U_z x - b_g)
        h = tanh(W_g y + U_g (r * x))
        g(x, y) = (1 - z) * x + z * h

    The residual stream x plays the GRU's recurrent state and the sublayer
    output y plays its input.
    """
    def __init__(self, d_hidden:int, rngs:nnx.Rngs, bias_init:float):
        self.reset_y_linear = _linear(d_hidden, rngs)
        self.reset_x_linear = _linear(d_hidden, rngs)
        self.update_y_linear = _linear(d_hidden, rngs)
        self.update_x_linear = _linear(d_hidden, rngs)
        self.candidate_y_linear = _linear(d_hidden, rngs)
        self.candidate_x_linear = _linear(d_hidden, rngs)
        self.gate_bias = nnx.Param(jnp.full((d_hidden,), bias_init))

    def __call__(self, x, y):
        r = nnx.sigmoid(self.reset_y_linear(y) + self.reset_x_linear(x))
        z = nnx.sigmoid(self.update_y_linear(y) + self.update_x_linear(x) - self.gate_bias)
        h = nnx.tanh(self.candidate_y_linear(y) + self.candidate_x_linear(r * x))
        return (1 - z) * x + z * h

def make_gate(gating:str, d_hidden:int, rngs:nnx.Rngs, bias_init:float=2.0):
    """Build one gating layer, used in place of a residual add.

    "residual" is the ungated default and adds no parameters, so a block built
    with it is identical to an ungated block. bias_init is ignored by
    "residual" and "input", which have no b_g.
    """
    if gating == "residual":
        return ResidualGate()
    if gating == "input":
        return InputGate(d_hidden, rngs)
    if gating == "output":
        return OutputGate(d_hidden, rngs, bias_init)
    if gating == "highway":
        return HighwayGate(d_hidden, rngs, bias_init)
    if gating == "sigtanh":
        return SigTanhGate(d_hidden, rngs, bias_init)
    if gating == "gru":
        return GRUGate(d_hidden, rngs, bias_init)

    raise ValueError(f"unknown gating '{gating}'; expected one of {sorted(GATES)}")
