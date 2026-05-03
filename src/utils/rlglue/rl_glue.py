from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import utils.chex as cxu
from utils.rlglue.agent import BaseAgent
from utils.rlglue.environment import BaseEnvironment


@cxu.dataclass
class GlueState:
    agent_state: Any
    env_state: Any
    last_action: jax.Array | None
    num_episodes: int
    num_steps: int
    total_reward: float
    total_steps: int


@cxu.dataclass
class Interaction:
    obs: jax.Array
    action: jax.Array | None
    term: jax.Array
    trunc: jax.Array
    reward: jax.Array
    extra: dict[str, jax.Array]


class RlGlue:
    def __init__(self, agent: BaseAgent, env: BaseEnvironment):
        self.environment = env
        self.agent = agent

        self.state = GlueState(
            agent_state=agent.state,
            env_state=env.state,
            last_action=None,
            num_episodes=0,
            num_steps=0,
            total_reward=0.0,
            total_steps=0,
        )

    def start(self):
        self.state, (s, last_action) = self._start(self.state)
        return s, last_action

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: GlueState):
        state.env_state, s = self.environment._start(state.env_state)
        state.agent_state, state.last_action = self.agent._start(state.agent_state, s)
        return state, (s, state.last_action)

    def step(self) -> Interaction:
        assert self.state.last_action is not None, (
            "Action is None; make sure to call glue.start() before calling glue.step()."
        )
        self.state, interaction = self._step(self.state)
        return interaction

    @partial(jax.jit, static_argnums=0)
    def _step(self, state: GlueState):
        return self._step_with_agent_methods(state, self.agent._step, self.agent._end)

    @partial(jax.jit, static_argnums=0)
    def _step_without_update(self, state: GlueState):
        return self._step_with_agent_methods(
            state, self.agent._step_without_update, self.agent._end_without_update
        )

    def _step_with_agent_methods(self, state: GlueState, agent_step, agent_end):
        state.env_state, (s, reward, term, trunc, extra) = self.environment._step(
            state.env_state, state.last_action
        )

        state.total_reward += reward

        state.num_steps += 1
        state.total_steps += 1

        def _end_step(_state, _s, _reward, _term, _trunc, _extra):
            _state.num_episodes += 1
            _state.agent_state = agent_end(_state.agent_state, _reward, _extra)
            interaction = Interaction(
                obs=_s,
                action=jnp.full_like(_state.last_action, -1),
                term=_term,
                trunc=_trunc,
                reward=_reward,
                extra=_extra,
            )
            return _state, interaction

        def _normal_step(_state, _s, _reward, _term, _trunc, _extra):
            _state.agent_state, _state.last_action = agent_step(
                _state.agent_state, _reward, _s, _extra
            )
            interaction = Interaction(
                obs=_s,
                action=_state.last_action,
                term=_term,
                trunc=_trunc,
                reward=_reward,
                extra=_extra,
            )
            return _state, interaction

        state, interaction = jax.lax.cond(
            term,
            _end_step,
            _normal_step,
            state,
            s,
            reward,
            term,
            trunc,
            extra,
        )

        return state, interaction

    def runEpisode(self, max_steps: int = 0):
        is_terminal = False

        self.start()

        while (not is_terminal) and (
            (max_steps == 0) or (self.state.num_steps < max_steps)
        ):
            rl_step_result = self.step()
            is_terminal = rl_step_result.term

        # even at episode cutoff, this still counts as completing an episode
        if not is_terminal:
            self.num_episodes += 1

        return is_terminal
