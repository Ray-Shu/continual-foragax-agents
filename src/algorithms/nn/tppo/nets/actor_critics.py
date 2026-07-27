import distrax
from flax import nnx

class ActorHead(nnx.Module):
    def __init__(self, d_hidden, rngs):
        self.linear = nnx.Linear(in_features=d_hidden, out_features=4, rngs=rngs)

    def __call__(self, x):
        """
        Returns a distrax categorical distribution that can be sampled from.
        """
        logits = self.linear(x)
        actions = distrax.Categorical(logits=logits)
        return actions

class CriticHead(nnx.Module):
    def __init__(self, d_hidden, rngs):
        self.linear = nnx.Linear(in_features=d_hidden, out_features=1, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)
