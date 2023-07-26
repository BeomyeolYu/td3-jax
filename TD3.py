"""
pip install jax==0.4.13 jaxlib==0.4.13 flax==0.7.0
"""

import numpy as np
import jax # NN lib built on top of JAX developed by Google Research
from jax import numpy as jnp
from jax import random as jrandom
from flax import linen as nn
from flax import serialization
import optax # JAX optimizers - a separate lib developed by DeepMind
import utils

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):

    action_dim: int
    max_action: float

    def setup(self):
        self.l1 = nn.Dense(256)
        self.l2 = nn.Dense(256)
        self.l3 = nn.Dense(self.action_dim)

    #@linen.compact
    def __call__(self, state):
        a = nn.relu(self.l1(state))
        a = nn.relu(self.l2(a))
        action = self.max_action * nn.tanh(self.l3(a))
        return action


class Critic(nn.Module):

    def setup(self):
        # Q1 architecture
        self.l1 = nn.Dense(256)
        self.l2 = nn.Dense(256)
        self.l3 = nn.Dense(1)

        # Q2 architecture
        self.l4 = nn.Dense(256)
        self.l5 = nn.Dense(256)
        self.l6 = nn.Dense(1)

    #@linen.compact
    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        q1 = nn.relu(self.l1(sa))
        q1 = nn.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = nn.relu(self.l4(sa))
        q2 = nn.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        q1 = nn.relu(self.l1(sa))
        q1 = nn.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3:

    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005,
                policy_noise=0.2, noise_clip=0.5, policy_freq=2, seed=0, jit=False,
    ):  

        self.rngs = utils.PRNGKeys(seed)

        # initialize models
        # Generate (random) ground truth W and b
        # Note: we could get W, b from a randomely initialized nn.Dense here
        init_state  = jnp.zeros([1, state_dim],  dtype=jnp.float32)
        init_action = jnp.zeros([1, action_dim], dtype=jnp.float32)
        #key, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)

        # Create the actor networks and the optimizer:
        actor_rng_key = self.rngs.get_key()
        self.actor = Actor(action_dim, max_action)
        self.actor_params = self.actor.init(actor_rng_key, init_state)
        self.actor_target_params = self.actor.init(actor_rng_key, init_state)
        """
        self.actor_optimizer = optax.adamw(learning_rate=3e-4)
        """
        schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1.0,
        warmup_steps=50,
        decay_steps=1_000,
        end_value=3e-4,
        )
        self.actor_optimizer = optax.chain(
        optax.clip(1.0),
        optax.adamw(learning_rate=schedule),
        )
        self.actor_optimizer_state = self.actor_optimizer.init(self.actor_params)

        # Create the critic networks and the optimizer:
        critic_rng_key = self.rngs.get_key()
        self.critic = Critic()
        self.critic_params = self.critic.init(critic_rng_key, init_state, init_action)
        self.critic_target_params = self.critic.init(critic_rng_key, init_state, init_action) 
        """
        self.critic_optimizer = optax.adamw(learning_rate=3e-4)
        """
        self.critic_optimizer = optax.chain(
        optax.clip(1.0),
        optax.adamw(learning_rate=schedule),
        )
        self.critic_optimizer_state = self.critic_optimizer.init(self.critic_params)

        # Jit
        self.actor_jit = jax.jit(self.actor_jit)
        self.actor_update_step = jax.jit(self.actor_update_step)
        self.critic_update_step = jax.jit(self.critic_update_step)
        self.update_target_params = jax.jit(self.update_target_params)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def actor_jit(self, actor_params, state):
        return self.actor.apply(actor_params, state)


    def select_action(self, state):
        state = jax.device_put(state[None])
        return np.array(self.actor_jit(self.actor_params, state)).flatten()
    

    def get_actor_loss(self, actor_params, critic_params, state):
        # Set actor loss s.t. Q(s,\mu(s)) approximates \max_a Q(s,a):
        action = self.actor.apply(actor_params, state)
        Q_value = self.critic.apply(critic_params, state, action, method=self.critic.Q1)
        actor_loss = -jnp.mean(Q_value)
        return actor_loss
    

    def get_critic_loss(self, 
                        critic_params, 
                        critic_target_params, 
                        actor_target_params,
                        transition, 
                        rng):
        
        # Randomly sample a batch of transitions from an experience replay buffer:
        state, action, next_state, reward, not_done = transition

        # Add clipped noise to target actions for 'target policy smoothing':
        noise = jrandom.normal(rng, action.shape) * self.policy_noise
        noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)

        # Compute target actions from a target policy network:
        next_action = self.actor.apply(actor_target_params, next_state)
        next_action = jnp.clip(next_action + noise, -self.max_action, self.max_action)
        
        # Get target Q-values, Q_targ(s', a'): 
        target_Q1, target_Q2 = self.critic.apply(critic_target_params, next_state, next_action)

        # Use a smaller target Q-value:
        target_Q = jnp.minimum(target_Q1, target_Q2)

        # Compute targets, y(r, s', d):
        target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q-values, Q1(s, a) and Q2(s, a):
        current_Q1, current_Q2 = self.critic.apply(critic_params, state, action)

        # Set a mean-squared Bellman error (MSBE) loss function:
        Q1_loss = jnp.mean(jnp.square(current_Q1 - target_Q))
        Q2_loss = jnp.mean(jnp.square(current_Q2 - target_Q))
        critic_loss = Q1_loss + Q2_loss

        return critic_loss

    def critic_update_step(self,
                           critic_params, 
                           critic_target_params, 
                           actor_target_params, 
                           critic_optimizer_state,
                           transition, 
                           rng):

        critic_value_and_grad = jax.value_and_grad(self.get_critic_loss)
        critic_loss, critic_grad = critic_value_and_grad(
            critic_params,
            critic_target_params,
            actor_target_params,
            transition,
            rng
        )
        
        critic_params_update, critic_optimizer_state = self.critic_optimizer.update(
            critic_grad, critic_optimizer_state, critic_params
        )
        critic_params = optax.apply_updates(critic_params, critic_params_update)

        return critic_params, critic_optimizer_state, critic_loss
    
    def actor_update_step(self, 
                          actor_params, 
                          critic_params, 
                          actor_optimizer_state,
                          state):

        actor_value_and_grad = jax.value_and_grad(self.get_actor_loss)
        actor_loss, actor_grad = actor_value_and_grad(
            actor_params, 
            critic_params, 
            state
        )
        actor_params_update, actor_optimizer_state = self.actor_optimizer.update(
            actor_grad, actor_optimizer_state, actor_params
        )
        actor_params = optax.apply_updates(actor_params, actor_params_update)

        return actor_params, actor_optimizer_state, actor_loss

    def update_target_params(self, params, target_params):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param
        # updated_params = jax.tree_multimap(_update, params, target_params)
        updated_params = jax.tree_util.tree_map(_update, params, target_params)
        return updated_params

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        transition = (state, action, next_state, reward, not_done)

        # Critic step:
        critic_step_rng = self.rngs.get_key()
        self.critic_params, self.critic_optimizer_state, _ = self.critic_update_step(
                self.critic_params,
                self.critic_target_params,
                self.actor_target_params,
                self.critic_optimizer_state,
                transition,
                critic_step_rng)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Actor step:
            self.actor_params, self.actor_optimizer_state, _ = self.actor_update_step(
                self.actor_params,
                self.critic_params,
                self.actor_optimizer_state,
                state)

            # Update the frozen target models
            params = (self.actor_params, self.critic_params)
            target_params = (self.actor_target_params, self.critic_target_params)
            updated_params = self.update_target_params(params, target_params)
            self.actor_target_params, self.critic_target_params = updated_params

    def save(self, filename):
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'wb') as f:
            f.write(serialization.to_bytes(self.critic_params))
        actor_file = filename + 'actor.ckpt'
        with open(actor_file, 'wb') as f:
            f.write(serialization.to_bytes(self.actor_params))

    def load(self, filename):
        # TODO: model loading is untested
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'rb') as f:
            self.critic_params = serialization.from_bytes(self.critic_params, f.read())
        self.critic_target_params = self.critic_params
        actor_file = filename + 'actor.ckpt'
        with open(actor_file, 'rb') as f:
            self.actor_params = serialization.from_bytes(self.actor_params, f.read())
        self.actor_target_params = self.actor_params
        