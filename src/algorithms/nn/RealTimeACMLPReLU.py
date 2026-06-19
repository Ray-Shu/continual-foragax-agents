# ReLU-activation variant of RealTimeACMLP: every layer uses ReLU (the RTU
# output is already post-ReLU), and the plasticity probes are sown POST-ReLU so
# that dormancy (Sokar) is measured on the actual unit outputs at every layer.
# Pairs with _compute_plasticity_metrics_relu in rtu_ppo.py.
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp
import numpy as np
import distrax
from algorithms.nn.rtus.rtus import RTLRTUs, RTNLRTUs


class RealTimeActorCriticMLPReLU(nn.Module):
    action_dim: int
    d_hidden: int = 192
    hidden_size: int = 64
    activation: str = "relu"  # ignored; this variant is ReLU throughout
    cont: bool = False
    rtu_type: str = "linear_rtu"
    use_sinusoidal_encoding: bool = False
    use_reward_trace: bool = False
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, hidden, obs):
        activation = nn.relu

        if self.rtu_type == "linear_rtu":
            seq_model = RTLRTUs
        elif self.rtu_type == "non_linear_rtu":
            seq_model = RTNLRTUs
        else:
            raise NotImplementedError

        (actor_hidden, critic_hidden) = hidden
        (obs, last_action_encoded, last_reward, sine, cosine, reward_trace) = obs
        last_reward_plus = last_reward
        if self.use_sinusoidal_encoding:
            last_reward_plus = jnp.concatenate((last_reward_plus, sine, cosine), axis=-1)
        if self.use_reward_trace:
            last_reward_plus = jnp.concatenate((last_reward_plus, reward_trace), axis=-1)

        obs = jnp.reshape(obs, (obs.shape[0], -1))

        actor_embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0), name="actor_dense1",
        )(obs)
        if self.use_layernorm:
            actor_embedding = nn.LayerNorm(name="actor_layernorm1")(actor_embedding)
        actor_embedding = activation(actor_embedding)
        # Post-ReLU probe (dormancy is measured on the unit outputs).
        self.sow("intermediates", "actor_pre1", actor_embedding)
        actor_embedding = jnp.concatenate(
            (actor_embedding, last_action_encoded, last_reward_plus), axis=-1
        )
        actor_embedding_skip = actor_embedding

        critic_embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0), name="critic_dense1",
        )(obs)
        if self.use_layernorm:
            critic_embedding = nn.LayerNorm(name="critic_layernorm1")(critic_embedding)
        critic_embedding = activation(critic_embedding)
        self.sow("intermediates", "critic_pre1", critic_embedding)
        critic_embedding = jnp.concatenate(
            (critic_embedding, last_action_encoded, last_reward_plus), axis=-1
        )
        critic_embedding_skip = critic_embedding

        actor_hidden, actor_embedding = seq_model(self.d_hidden, params_type="exp_exp", name="actor_rtu")(actor_hidden, actor_embedding)
        critic_hidden, critic_embedding = seq_model(self.d_hidden, params_type="exp_exp", name="critic_rtu")(critic_hidden, critic_embedding)
        # RTU output is already post-ReLU (default activation in RTLRTUs/RTNLRTUs).
        self.sow("intermediates", "actor_rtu_out", actor_embedding)
        self.sow("intermediates", "critic_rtu_out", critic_embedding)
        actor_embedding = jnp.concatenate((actor_embedding, actor_embedding_skip), axis=-1)
        critic_embedding = jnp.concatenate((critic_embedding, critic_embedding_skip), axis=-1)

        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor_dense2")(actor_embedding)
        if self.use_layernorm:
            actor_mean = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm2")(actor_mean)
        actor_mean = activation(actor_mean)
        self.sow("intermediates", "actor_pre2", actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0), name="actor_mean",
        )(actor_mean)
        if self.cont:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(2),
            bias_init=constant(0.0), name="critic_dense2",
        )(critic_embedding)
        if self.use_layernorm:
            critic = nn.LayerNorm(name="critic_layernorm2")(critic)
        critic = activation(critic)
        self.sow("intermediates", "critic_pre2", critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value"
        )(critic)
        hidden = (actor_hidden, critic_hidden)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input):
        return RealTimeActorCriticMLPReLU._init(batch_size, d_hidden, d_input)

    @staticmethod
    def _init(batch_size, d_hidden, d_input):
        def side():
            hidden_init = (jnp.zeros((batch_size, d_hidden)), jnp.zeros((batch_size, d_hidden)))
            memory_grad_init = (
                jnp.zeros((batch_size, d_hidden)), jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_hidden)), jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_input, d_hidden)), jnp.zeros((batch_size, d_input, d_hidden)),
                jnp.zeros((batch_size, d_input, d_hidden)), jnp.zeros((batch_size, d_input, d_hidden)),
            )
            return (hidden_init, memory_grad_init)
        return (side(), side())
