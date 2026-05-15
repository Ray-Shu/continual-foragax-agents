from functools import partial
from typing import Callable, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import jax.tree
import optax


def compute_utility(
    p: chex.Array, grad: chex.Array, key: chex.PRNGKey, utility_name: str
) -> tuple[chex.Array, chex.PRNGKey]:
    """Compute utility of parameters for reinitialization."""
    if utility_name == "gradient":
        return jnp.abs(grad * p), key
    elif utility_name == "magnitude":
        return jnp.abs(p), key
    elif utility_name == "random":
        key, subkey = jax.random.split(key)
        return jax.random.uniform(subkey, shape=p.shape), key
    else:
        raise ValueError(f"Utility function not recognized: {utility_name}")


def prune_weights(
    utility: chex.Array, pruning_method: str, reinit_factor: float, key: chex.PRNGKey
) -> chex.Array:
    """Returns a boolean mask indicating which weights should be pruned.

    Returns a boolean array of the same size as utility, where True indicates
    a weight that should be reinitialized.
    """
    if pruning_method == "proportional":
        fraction_to_prune = utility.size * reinit_factor
        key, subkey = jax.random.split(key)
        drop_num = int(fraction_to_prune) + jax.random.bernoulli(
            subkey, fraction_to_prune % 1
        ).astype(jnp.int32)

        # Get sorted indices (lowest utility first)
        indices = jnp.argsort(utility)

        # Create mask: True for weights to reinitialize (lowest utility)
        mask = jnp.arange(utility.size) < drop_num

        # Map back to original parameter positions
        reinit_mask = jnp.zeros(utility.shape, dtype=bool)
        reinit_mask = reinit_mask.at[indices].set(mask)

        return reinit_mask

    elif pruning_method == "threshold":
        prune_threshold = reinit_factor * jnp.mean(utility)
        return utility <= prune_threshold

    else:
        raise ValueError(f"Pruning method not recognized: {pruning_method}")


class SWRState(NamedTuple):
    """State for SelectiveWeightReinitialization optimizer."""

    step: chex.Array
    avg_utility: optax.Params
    rng_key: chex.PRNGKey
    reinit_indicator: chex.Array
    num_replaced: chex.Array


def selective_weight_reinitialization(
    utility_function: str,
    pruning_method: str,
    initializers: dict[str, Callable],
    reinit_freq: int = 0,
    reinit_factor: float = 0.0,
    decay_rate: float = 0.0,
    seed: int = 0,
) -> optax.GradientTransformation:
    """
    Optax GradientTransformation for Selective Weight Reinitialization.

    Arguments:
        utility_function: str in ["gradient", "magnitude", "random"]
        pruning_method: str in ["proportional", "threshold"]
        initializers: dict mapping parameter names to Haiku initializer callables
        reinit_freq: how often to reinitialize (in steps)
        reinit_factor: fraction/threshold for reinitialization
        decay_rate: exponential moving average decay for utility
        seed: random seed for reinitialization
    """

    def init_fn(params: optax.Params) -> SWRState:
        """Initialize optimizer state."""
        return SWRState(
            step=jnp.array(0, dtype=jnp.int32),
            avg_utility=jax.tree.map(lambda p: jnp.zeros(p.shape), params),
            rng_key=jax.random.PRNGKey(seed),
            reinit_indicator=jnp.array(False),
            num_replaced=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: optax.Updates,
        state: SWRState,
        params: Optional[optax.Params] = None,
        grad: Optional[optax.Updates] = None,
    ) -> tuple[optax.Updates, SWRState]:
        """Update parameters with selective reinitialization."""
        new_step = state.step + 1

        # Update moving average of utility
        def decay_avg_utility(avg_utility, params, grad):
            return jax.tree.map(
                lambda avg, p, g: (
                    decay_rate * avg
                    + (1.0 - decay_rate)
                    * compute_utility(p, g, state.rng_key, utility_function)[0]
                ),
                avg_utility,
                params,
                grad,
            )

        new_avg_utility = jax.lax.cond(
            decay_rate > 0.0,
            decay_avg_utility,
            lambda avg_utility, params, grad: avg_utility,
            state.avg_utility,
            params,
            grad,
        )

        # Check if it's time to reinitialize
        should_reinit = (reinit_freq > 0) & (new_step % reinit_freq == 0)

        def do_reinit(carry):
            """Perform reinitialization."""
            params, updates, key = carry

            num_params = len(jax.tree_util.tree_leaves(params))
            keys = jax.random.split(key, num_params + 1)
            key = keys[0]
            param_keys = keys[1:]

            keys_tree = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(params), param_keys
            )

            def process_single_param(p, update, g, avg_utility, initializer, param_key):
                utility_key, prune_key, reinit_key = jax.random.split(param_key, 3)

                utility, utility_key = jax.lax.cond(
                    decay_rate > 0.0,
                    lambda *_: (avg_utility, utility_key),
                    partial(compute_utility, utility_name=utility_function),
                    p,
                    g,
                    utility_key,
                )

                reinit_mask = prune_weights(
                    utility, pruning_method, reinit_factor, prune_key
                )

                # Count number of weights to reinitialize
                num_reinit = jnp.sum(reinit_mask)

                # Generate new values for positions to reinitialize using the initializer
                new_values = initializer(reinit_key, p.shape, p.dtype)

                # Update parameters using the mask
                new_update = jnp.where(reinit_mask, new_values - p, update)

                new_avg_u = jax.lax.cond(
                    decay_rate > 0,
                    # Reset utility for reinitialized weights
                    lambda avg_utility: jnp.where(reinit_mask, 0.0, avg_utility),
                    lambda avg_utility: avg_utility,
                    avg_utility,
                )

                return new_update, new_avg_u, num_reinit

            results_tree = jax.tree.map(
                process_single_param,
                params,
                updates,
                grad,
                new_avg_utility,
                initializers,
                keys_tree,
            )

            tuple_of_trees = jax.tree_util.tree_transpose(
                jax.tree_util.tree_structure(params),
                jax.tree_util.tree_structure((0, 0, 0)),
                results_tree,
            )
            updates, new_avg_utility2, num_reinit_tree = tuple_of_trees

            total_replaced = jax.tree_util.tree_reduce(
                lambda x, y: x + y, num_reinit_tree, initializer=0
            )
            return updates, new_avg_utility2, key, total_replaced

        def no_reinit(carry):
            """Skip reinitialization."""
            params, updates, key = carry
            return updates, new_avg_utility, key, 0

        new_updates, new_avg_utility, new_key, num_replaced = jax.lax.cond(
            should_reinit,
            do_reinit,
            no_reinit,
            (params, updates, state.rng_key),
        )

        new_state = SWRState(
            step=new_step,
            avg_utility=new_avg_utility,
            rng_key=new_key,
            reinit_indicator=jnp.array(should_reinit),
            num_replaced=num_replaced,
        )

        return new_updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)  # type: ignore


class UPGDState(NamedTuple): 
    """State for Utility based Perturbed Gradient Descent optimizer."""
    step: chex.Array 
    utility_history: optax.Params
    rng_key: chex.PRNGKey 


def upgd_optimizer(
    step_size: float, 
    utility_decay_rate: float, 
    noise: float, 
    utility: float, 
    seed: int = 0,

)-> optax.GradientTransformation:
    """Optax GradientTransformation for Utility based Perturbed Gradient Descent.

    Arguments:
        step_size: the learning rate alpha
        utility_decay_rate: the decay rate of ulity beta 
        nosie: gaussian noise added to the gradient 
        utility: the utility measurement U: [0, 1]
        seed: random seed for reinitialization
    """
    
    def init_fn(params):
        """Called ONCE at the start. params is your network weights.
        Returns: initial state object
        """
        return MyState(
            step=0,
            utility=jax.tree.map(lambda p: jnp.zeros(p.shape), params),
            key=jax.random.PRNGKey(seed),
        )
    
    def update_fn(updates, state, params=None, grad=None):
        """Called EVERY training step. 
        Args:
          updates: the gradients coming FROM the previous optimizer in the chain (or raw gradients)
          state: current optimizer state
          params: the CURRENT network weights (optional, but UPGD needs this)
          grad: the ORIGINAL gradients (optional, but UPGD needs this for utility)
        
        Returns: (new_updates, new_state)
          new_updates: transformed gradients to pass to next optimizer
          new_state: updated state object
        """
        new_step = state.step + 1
        
        # Transform the updates/gradients here
        # Use grad and params to compute utility
        # Add noise
        # Weight by utility
        # Return modified updates
        
        new_state = MyState(
            step=new_step,
            utility=new_utility,
            key=new_key,
        )
        return new_updates, new_state
    
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)