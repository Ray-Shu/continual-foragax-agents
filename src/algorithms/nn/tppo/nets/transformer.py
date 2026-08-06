import jax
import jax.numpy as jnp
from flax import nnx

from algorithms.nn.tppo.nets.transformer_utils.activations import make_ffn
from algorithms.nn.tppo.nets.transformer_utils.gating import make_gate

class TransformerBlock(nnx.Module):
    def __init__(self, T:int, d_hidden:int, d_keys:int, d_vals:int, d_ff:int, rngs:nnx.Rngs, band:int|None=None, activation:str="relu", gating:str="residual", gate_bias_init:float=2.0):
        self.context_window = T
        self.band = T if band is None else band  # band is how much the model sees (ie. receptive field)
        self.d_hidden = d_hidden
        self.d_queries = d_keys  # d_queries must be the same as d_keys
        self.d_keys = d_keys
        self.d_values = d_vals
        self.d_ff = d_ff
        self.activation = activation
        self.gating = gating
        self.rngs = rngs
        self.relative_bias = nnx.Param(jnp.zeros((self.band))) # positional encodings relative to each transformer block

        self.layernorm1 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)
        self.layernorm2 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)

        self.queries_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_queries, rngs=self.rngs)
        self.keys_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_keys, rngs=self.rngs)
        self.values_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.d_values, rngs=self.rngs)
        self.output_linear = nnx.Linear(in_features=self.d_values, out_features=self.d_hidden, rngs=self.rngs)

        self.ffn = make_ffn(self.activation, self.d_hidden, self.d_ff, self.rngs)

        self.attn_gate = make_gate(self.gating, self.d_hidden, self.rngs, gate_bias_init)
        self.ffn_gate = make_gate(self.gating, self.d_hidden, self.rngs, gate_bias_init)

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
        x2 = self.attn_gate(x, self.output_linear(attn_out))

        x2_ln = self.layernorm2(x2)
        return self.ffn_gate(x2, self.ffn(x2_ln))
