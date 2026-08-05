import jax
import jax.numpy as jnp
from flax import nnx

# pointwise activations usable in the (2-matrix) feedforward network
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

class TransformerBlock(nnx.Module):
    def __init__(self, T:int, d_hidden:int, d_keys:int, d_vals:int, d_ff:int, rngs:nnx.Rngs, band:int|None=None, activation:str="relu"):
        self.context_window = T
        self.band = T if band is None else band  # band is how much the model sees (ie. receptive field)
        self.d_hidden = d_hidden
        self.d_queries = d_keys  # d_queries must be the same as d_keys
        self.d_keys = d_keys
        self.d_values = d_vals
        self.d_ff = d_ff
        self.activation = activation
        self.rngs = rngs
        self.relative_bias = nnx.Param(jnp.zeros((self.band))) # positional encodings relative to each transformer block

        self.layernorm1 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)
        self.layernorm2 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)

        self.queries_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_queries, rngs=self.rngs)
        self.keys_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_keys, rngs=self.rngs)
        self.values_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_values, rngs=self.rngs)
        self.output_linear = nnx.Linear(in_features=self.d_values, out_features=self.d_hidden, rngs=self.rngs)

        self.ffn = make_ffn(self.activation, self.d_hidden, self.d_ff, self.rngs)

    def __call__(self, x):
        M = x.shape[1]
        x_ln = self.layernorm1(x)

        # attn block
        Q = self.queries_linear(x_ln)
        K = self.keys_linear(x_ln)
        V = self.values_linear(x_ln)

        # create a context-window aware mask
        i = jnp.arange(M)[:, None]
        j = jnp.arange(M)[None, :]
        causal = (j <= i) & (j > i - self.band)

        dist = jnp.clip(i - j, 0, self.band - 1)   # 0 = self, band-1 = oldest in window

        scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(self.d_keys)
        scores = scores + self.relative_bias[dist]
        scores = jnp.where(causal, scores, -jnp.inf)

        attn_out = jnp.matmul(nnx.softmax(scores, axis=-1), V)
        x2 = self.output_linear(attn_out) + x

        x2_ln = self.layernorm2(x2)
        return self.ffn(x2_ln) + x2