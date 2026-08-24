import jax
import jax.numpy as jnp
from flax import nnx

from algorithms.nn.tppo.nets.transformer_utils.activations import make_ffn
from algorithms.nn.tppo.nets.transformer_utils.gating import make_gate

def _sinusoidal_embedding(distances, dim): 
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    sinusoid = jnp.outer(distances, inv_freq)
    return jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)

class TransformerBlock(nnx.Module):
    def __init__(self, memory_len:int, d_hidden:int, d_keys:int, d_vals:int, d_ff:int, rngs:nnx.Rngs, num_query_heads:int=1, num_kv_heads:int=1, activation:str="relu", gating:str="residual", gate_bias_init:float=2.0):
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be a multiple of num_kv_heads"
        self.memory_len = memory_len # size of the stop-gradient memory buffer 
        self.d_hidden = d_hidden
        self.d_queries = d_keys  # per-head dim; d_queries is the same as d_keys
        self.d_keys = d_keys
        self.d_values = d_vals
        self.d_ff = d_ff
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_query_heads // num_kv_heads  # query heads sharing each kv head
        self.activation = activation
        self.gating = gating
        self.rngs = rngs

        self.pos_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.num_query_heads * self.d_keys, rngs=self.rngs)
        self.u = nnx.Param(jnp.zeros((self.num_query_heads, self.d_keys))) # content bias
        self.v = nnx.Param(jnp.zeros((self.num_query_heads, self.d_keys))) # position bias 

        self.layernorm1 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)
        self.layernorm2 = nnx.LayerNorm(num_features=self.d_hidden, rngs=self.rngs)

        self.queries_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.num_query_heads * self.d_queries, rngs=self.rngs)
        self.keys_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.num_kv_heads * self.d_keys, rngs=self.rngs)
        self.values_linear = nnx.Linear(in_features=self.d_hidden, out_features=self.num_kv_heads * self.d_values, rngs=self.rngs)
        self.output_linear = nnx.Linear(in_features=self.num_query_heads * self.d_values, out_features=self.d_hidden, rngs=self.rngs)

        self.ffn = make_ffn(self.activation, self.d_hidden, self.d_ff, self.rngs)

        self.attn_gate = make_gate(self.gating, self.d_hidden, self.rngs, gate_bias_init)
        self.ffn_gate = make_gate(self.gating, self.d_hidden, self.rngs, gate_bias_init)

    def __call__(self, x, memory):
        """
        x: (B, N, d_hidden). N = rollout size
        memory: (B, memory_len, d_hidden)

        Returns (out, new_memory) 
        """
        B, N, _ = x.shape
        full_input = jnp.concatenate([memory, x], axis=1)
        full_seq = self.memory_len + N 
        full_ln = self.layernorm1(full_input)

        # attn block
        Q = self.queries_linear(x).reshape(B, N, self.num_query_heads, self.d_queries)
        K = self.keys_linear(full_ln).reshape(B, full_seq, self.num_kv_heads, self.d_keys)
        V = self.values_linear(full_ln).reshape(B, full_seq, self.num_kv_heads, self.d_values)

        # share each kv head across its group of query heads
        K = jnp.repeat(K, self.group_size, axis=2)  # (B, M, num_query_heads, d_keys)
        V = jnp.repeat(V, self.group_size, axis=2)  # (B, M, num_query_heads, d_values)

        # move heads next to batch so matmul below is batched per-head: (B, H, M, d)
        Q = jnp.moveaxis(Q, 2, 1)
        K = jnp.moveaxis(K, 2, 1)
        V = jnp.moveaxis(V, 2, 1)

        # create a context-window aware mask
        i = jnp.arange(N)[:, None] + self.memory_len
        j = jnp.arange(full_seq)[None, :]
        causal = (j <= i)

        dist = jnp.clip(i - j, 0, full_seq - 1)  

        position_embedding = _sinusoidal_embedding(jnp.arange(full_seq, dtype=jnp.float32), self.d_hidden) 
        r = self.pos_linear(position_embedding).reshape(full_seq, self.num_query_heads, self.d_keys)
        r = r[dist]

        Q_u = Q + self.u[None, :, None, :] 
        Q_v = Q + self.v[None, :, None, :]

        content_attn = jnp.matmul(Q_u, jnp.swapaxes(K, -1, -2))
        position_attn = jnp.einsum('bhik,ijhk->bhij', Q_v, r)

        scores = (content_attn + position_attn) / jnp.sqrt(self.d_keys)
        scores = jnp.where(causal, scores, -jnp.inf)

        attn_out = jnp.matmul(nnx.softmax(scores, axis=-1), V)  # (B, H, M, d_values)
        attn_out = jnp.moveaxis(attn_out, 1, 2).reshape(B, N, self.num_query_heads * self.d_values)
        x2 = self.attn_gate(x, self.output_linear(attn_out))

        x2_ln = self.layernorm2(x2)
        out = self.ffn_gate(x2, self.ffn(x2_ln))

        new_memory = jax.lax.stop_gradient(full_input[:, -self.memory_len:])
        return out, new_memory
