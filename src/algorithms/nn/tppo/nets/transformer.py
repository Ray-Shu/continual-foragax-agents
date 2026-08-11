import jax
import jax.numpy as jnp
from flax import nnx

from algorithms.nn.tppo.nets.transformer_utils.activations import make_ffn
from algorithms.nn.tppo.nets.transformer_utils.gating import make_gate

class TransformerBlock(nnx.Module):
    def __init__(self, T:int, d_hidden:int, d_keys:int, d_vals:int, d_ff:int, rngs:nnx.Rngs, band:int|None=None, num_query_heads:int=1, num_kv_heads:int=1, activation:str="relu", gating:str="residual", gate_bias_init:float=2.0):
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be a multiple of num_kv_heads"
        self.context_window = T
        self.band = T if band is None else band  # band is how much the model sees (ie. receptive field)
        self.d_hidden = d_hidden
        self.d_queries = d_keys  # per-head dim; d_queries must be the same as d_keys
        self.d_keys = d_keys
        self.d_values = d_vals
        self.d_ff = d_ff
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_query_heads // num_kv_heads  # query heads sharing each kv head
        self.activation = activation
        self.gating = gating
        self.rngs = rngs
        self.relative_bias = nnx.Param(jnp.zeros((self.band))) # positional encodings relative to each transformer block

        self.layernorm1 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)
        self.layernorm2 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)

        self.queries_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.num_query_heads * self.d_queries, rngs=self.rngs)
        self.keys_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.num_kv_heads * self.d_keys, rngs=self.rngs)
        self.values_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.num_kv_heads * self.d_values, rngs=self.rngs)
        self.output_linear = nnx.Linear(in_features=self.num_query_heads * self.d_values, out_features=self.d_hidden, rngs=self.rngs)

        self.ffn = make_ffn(self.activation, self.d_hidden, self.d_ff, self.rngs)

        self.attn_gate = make_gate(self.gating, self.d_hidden, self.rngs, gate_bias_init)
        self.ffn_gate = make_gate(self.gating, self.d_hidden, self.rngs, gate_bias_init)

    def __call__(self, x):
        B, M, _ = x.shape
        x_ln = self.layernorm1(x)

        # attn block
        Q = self.queries_linear(x_ln).reshape(B, M, self.num_query_heads, self.d_queries)
        K = self.keys_linear(x_ln).reshape(B, M, self.num_kv_heads, self.d_keys)
        V = self.values_linear(x_ln).reshape(B, M, self.num_kv_heads, self.d_values)

        # share each kv head across its group of query heads
        K = jnp.repeat(K, self.group_size, axis=2)  # (B, M, num_query_heads, d_keys)
        V = jnp.repeat(V, self.group_size, axis=2)  # (B, M, num_query_heads, d_values)

        # move heads next to batch so matmul below is batched per-head: (B, H, M, d)
        Q = jnp.moveaxis(Q, 2, 1)
        K = jnp.moveaxis(K, 2, 1)
        V = jnp.moveaxis(V, 2, 1)

        # create a context-window aware mask
        i = jnp.arange(M)[:, None]
        j = jnp.arange(M)[None, :]
        causal = (j <= i) & (j > i - self.band)

        dist = jnp.clip(i - j, 0, self.band - 1)   # 0 = self, band-1 = oldest in window

        scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(self.d_keys)  # (B, H, M, M)
        scores = scores + self.relative_bias[dist]
        scores = jnp.where(causal, scores, -jnp.inf)

        attn_out = jnp.matmul(nnx.softmax(scores, axis=-1), V)  # (B, H, M, d_values)
        attn_out = jnp.moveaxis(attn_out, 1, 2).reshape(B, M, self.num_query_heads * self.d_values)
        x2 = self.attn_gate(x, self.output_linear(attn_out))

        x2_ln = self.layernorm2(x2)
        return self.ffn_gate(x2, self.ffn(x2_ln))
