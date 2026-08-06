from flax import nnx

# pointwise activations usable in the ffn
ACTIVATIONS = {
    "relu": nnx.relu,
    "gelu": nnx.gelu,
    "silu": nnx.silu,
    "swish": nnx.swish,
    "tanh": nnx.tanh,
    "elu": nnx.elu,
    "leaky_relu": nnx.leaky_relu,
}

class SwiGLU(nnx.Module):
    """
    Gated feedforward network: FFN(x) = W_down(silu(W_gate x) * W_up x).
    """
    def __init__(self, d_hidden:int, d_ff:int, rngs:nnx.Rngs):
        self.gate_linear = nnx.Linear(in_features=d_hidden, out_features=d_ff, rngs=rngs)
        self.up_linear = nnx.Linear(in_features=d_hidden, out_features=d_ff, rngs=rngs)
        self.down_linear = nnx.Linear(in_features=d_ff, out_features=d_hidden, rngs=rngs)

    def __call__(self, x):
        return self.down_linear(nnx.silu(self.gate_linear(x)) * self.up_linear(x))

def make_ffn(activation:str, d_hidden:int, d_ff:int, rngs:nnx.Rngs):
    """Build a transformer block's feedforward network for activation.

    "swiglu" is gated (see SwiGLU); any other name gives the usual
    Linear -> activation -> Linear.
    """
    if activation == "swiglu":
        return SwiGLU(d_hidden, d_ff, rngs)

    if activation not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation '{activation}'; expected 'swiglu' or one of "
            f"{sorted(ACTIVATIONS)}"
        )

    return nnx.Sequential(
        nnx.Linear(in_features=d_hidden, out_features=d_ff, rngs=rngs),
        ACTIVATIONS[activation],
        nnx.Linear(in_features=d_ff, out_features=d_hidden, rngs=rngs)
    )
