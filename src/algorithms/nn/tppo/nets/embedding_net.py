from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

class EmbeddingNet(nnx.Module):
    linear: nnx.Linear | None

    def __init__(self, d_hidden:int, rngs:nnx.Rngs):
        self.rngs = rngs
        self.d_hidden = d_hidden

        self.is_initialized = False
        self.linear = nnx.data(None)

    def __call__(self, obs):
        """
        obs: ((B, M, obs_dim), (B, M, action_dim), (B, M, 1)) where:
            - B is the batch size
            - M is the sequence length = T-1+N
            - T is context length, N is the rollout steps

        returns:
            E: embedding matrix of shape (B, M, d_hidden)
        """

        # concat alongside last dimension
        obs, last_action, last_reward = obs
        obs = jnp.reshape(obs, obs.shape[:2] + (-1,))
        U = jnp.concatenate(
            (obs, last_action, last_reward),
            axis = -1
        )

        if not self.is_initialized:
            # trainable embedding matrix (E)
            self.linear = nnx.Linear(
                in_features = U.shape[-1],
                out_features = self.d_hidden,
                rngs = self.rngs
            )
            self.is_initialized = True

        E = self.linear(U)

        return E