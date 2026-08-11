from typing import Any

import jax
import jax.numpy as jnp
import distrax
from flax import nnx

from algorithms.nn.tppo.nets.actor_critics import ActorHead, CriticHead
from algorithms.nn.tppo.nets.embedding_net import EmbeddingNet
from algorithms.nn.tppo.nets.transformer import TransformerBlock

class TransformerPPO(nnx.Module):
    def __init__(self, T:int, d_hidden:int, d_keys:int, d_vals:int, d_ff: int, rngs:nnx.Rngs, band:int|None = None, num_layers=1, num_query_heads:int=1, num_kv_heads:int=1, activation:str="relu", gating:str="residual", gate_bias_init:float=2.0):
        """
        T: size of the context window
        d_hidden: hidden dim
        d_keys, d_vals: per-head dimensions for queries/keys and values for the attention mechanism
        d_ff: dimension of the feedforward network in transformer block
        band: the "receptive field" of the transformer. Similar to context length, but works with stacks of transformer blocks
        num_query_heads, num_kv_heads: number of attention heads for queries and for keys/values.
        activation: type of activation function (ie. relu, gelu, elu, swiglu, etc.)
        gating: what replaces each sublayer's residual add -- "residual" (ungated,
            adds no parameters) or a GTrXL gate from nets.transformer_utils.gating.GATES
        gate_bias_init: b_g for the gates that have one; larger starts the gate
            closer to an identity map. Ignored when gating is "residual"/"input"
        """
        # params
        self.T = T
        self.d_hidden = d_hidden
        self.d_keys = d_keys
        self.d_vals = d_vals
        self.d_ff = d_ff
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.activation = activation
        self.gating = gating

        # nets
        self.embed = EmbeddingNet(self.d_hidden, rngs)
        self.actor = ActorHead(self.d_hidden, rngs)
        self.critic = CriticHead(self.d_hidden, rngs)

        self.blocks = nnx.List([
            TransformerBlock(self.T, self.d_hidden, self.d_keys, self.d_vals, self.d_ff, rngs, band=band, num_query_heads=self.num_query_heads, num_kv_heads=self.num_kv_heads, activation=self.activation, gating=self.gating, gate_bias_init=gate_bias_init)
            for _ in range(num_layers)
        ])


    def __call__(self, hidden:Any, obs, read_from=None):
        """
        hidden: Any
        obs: ((B, M, obs_dim), (B, M, action_dim), (B, M, 1)) where:
            - B is the batch size
            - M is the sequence length: T-1 + N, where T is the context window and N is the rollout buffer size

        read_from: which positions become decisions.
            None: last position only, (B, .)
            int: many decisions per sequence
        """
        x = self.embed(obs)

        for block in self.blocks:
            x = block(x)

        h = x[:, -1, :] if read_from is None else x[:, read_from:, :]
        pi, value = self.actor(h), self.critic(h)

        return pi, value