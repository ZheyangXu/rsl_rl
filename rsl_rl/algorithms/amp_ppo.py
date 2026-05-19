# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils.motion_loader import AmpLoader


class AmpPPO:
    """Proximal Policy Optimization with Adversarial Motion Priors (AMP).

    Combines PPO policy optimization with an adversarial discriminator that
    provides a style reward encouraging the policy to produce motions resembling
    a reference dataset (AMP paper, Peng et al. 2021).

    The discriminator is trained jointly with the policy using LSGAN loss,
    and a replay buffer prevents discriminator overfitting to recent policy data.

    Args:
        actor: The actor (policy) model.
        critic: The critic (value) model.
        discriminator: The AMP discriminator network.
        amp_data: AmpLoader providing expert motion data.
        storage: Rollout storage for PPO transitions.
        amp_replay_buffer_size: Capacity of the replay buffer for
            policy-generated AMP transitions.
        num_learning_epochs: PPO epochs per update.
        num_mini_batches: Mini-batches per epoch.
        clip_param: PPO clipping parameter.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        value_loss_coef: Value function loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        learning_rate: Initial learning rate (Adam).
        max_grad_norm: Maximum gradient norm for clipping.
        use_clipped_value_loss: Whether to use clipped value loss.
        schedule: Learning rate schedule ("fixed" or "adaptive").
        desired_kl: Target KL divergence for adaptive schedule.
        device: Torch device.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        discriminator: Discriminator,
        amp_data: AmpLoader,
        storage: RolloutStorage,
        amp_replay_buffer_size: int = 100000,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.device = device

        # Actor and critic
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Discriminator and AMP components
        self.discriminator = discriminator.to(self.device)
        obs_dim = self.discriminator.input_dim // 2
        self.amp_storage = ReplayBuffer(
            obs_dim=obs_dim, buffer_size=amp_replay_buffer_size, device=device
        )
        self.amp_data = amp_data

        # Optimizer: separate parameter groups for discriminator with weight decay
        optimizer_params = [
            {"params": self.actor.parameters(), "name": "actor"},
            {"params": self.critic.parameters(), "name": "critic"},
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": 1e-4,
                "name": "amp_trunk",
            },
            {
                "params": self.discriminator.linear.parameters(),
                "weight_decay": 1e-2,
                "name": "amp_head",
            },
        ]
        self.optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)

        # Rollout storage and transition
        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.amp_transition = RolloutStorage.Transition()

        # PPO hyperparameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data.

        Args:
            obs: TensorDict observation from the environment.

        Returns:
            Sampled action tensor.
        """
        self.transition.hidden_states = (
            self.actor.get_hidden_state(),
            self.critic.get_hidden_state(),
        )
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(
            self.transition.actions
        ).detach()
        self.transition.distribution_params = tuple(
            p.detach() for p in self.actor.output_distribution_params
        )
        self.transition.observations = obs
        return self.transition.actions

    def act_amp(self, amp_obs: torch.Tensor) -> None:
        """Store the current AMP observation for the replay buffer.

        Args:
            amp_obs: Flattened AMP observation tensor (B, obs_dim).
        """
        self.amp_transition.observations = amp_obs

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record one environment step and update normalizers.

        Args:
            obs: Observation from the environment after stepping.
            rewards: Combined (task + style) reward tensor.
            dones: Episode termination flags.
            extras: Additional metadata (e.g., time_outs).
        """
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def process_amp_step(self, next_amp_obs: torch.Tensor) -> None:
        """Insert a policy-generated AMP transition into the replay buffer.

        Args:
            next_amp_obs: The next AMP observation (B, obs_dim).
        """
        self.amp_storage.insert(self.amp_transition.observations, next_amp_obs)
        self.amp_transition.clear()

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE-lambda returns and advantages from stored transitions.

        Args:
            obs: Last observation from the rollout.
        """
        st = self.storage
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        self.critic.reset(hidden_state=critic_hidden_state)

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = (
                last_values if step == st.num_transitions_per_env - 1
                else st.values[step + 1]
            )
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = (
                st.rewards[step]
                + next_is_not_terminal * self.gamma * next_values
                - st.values[step]
            )
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        st.advantages = st.returns - st.values
        st.advantages = (st.advantages - st.advantages.mean()) / (
            st.advantages.std() + 1e-8
        )

    def update(self) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """Perform PPO + AMP discriminator update over stored batches.

        Returns:
            Tuple of 9 mean metrics:
            (value_loss, surrogate_loss, amp_loss, grad_pen_loss,
             policy_pred, expert_pred, accuracy_policy, accuracy_expert,
             kl_divergence).
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_accuracy_policy = 0.0
        mean_accuracy_expert = 0.0
        mean_accuracy_policy_elem = 0
        mean_accuracy_expert_elem = 0
        mean_kl_divergence = 0.0

        # Mini-batch generators
        is_recurrent = self.actor.is_recurrent or self.critic.is_recurrent
        if is_recurrent:
            ppo_generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            ppo_generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
            mini_batch_size=(
                self.storage.num_transitions_per_env
                * self.storage.num_envs
                // self.num_mini_batches
            ),
            allow_replacement=True,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_transitions_per_env
            * self.storage.num_envs
            // self.num_mini_batches,
        )

        for batch, sample_amp_policy, sample_amp_expert in zip(
            ppo_generator, amp_policy_generator, amp_expert_generator
        ):
            # --- Unpack PPO batch ---
            obs_batch = batch.observations
            actions_batch = batch.actions
            target_values_batch = batch.values
            advantages_batch = batch.advantages
            returns_batch = batch.returns
            old_actions_log_prob_batch = batch.old_actions_log_prob
            old_distribution_params = batch.old_distribution_params
            hidden_states_batch = batch.hidden_states
            masks_batch = batch.masks

            hidden_actor, hidden_critic = (None, None)
            if hidden_states_batch is not None:
                hidden_actor, hidden_critic = hidden_states_batch

            # --- Forward passes ---
            self.actor(
                obs_batch,
                masks=masks_batch,
                hidden_state=hidden_actor,
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(actions_batch)
            values = self.critic(
                obs_batch,
                masks=masks_batch,
                hidden_state=hidden_critic,
            )
            dist_params = self.actor.output_distribution_params
            entropy_batch = self.actor.output_entropy

            # --- Adaptive learning rate ---
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        old_distribution_params, dist_params
                    )
                    kl_mean = torch.mean(kl)
                    mean_kl_divergence += kl_mean.item()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # --- PPO surrogate loss ---
            ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # --- Value loss ---
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    values - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - values).pow(2).mean()

            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # --- AMP discriminator loss ---
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            policy_state = policy_state.to(self.device)
            policy_next_state = policy_next_state.to(self.device)
            expert_state = expert_state.to(self.device)
            expert_next_state = expert_next_state.to(self.device)

            policy_state_raw = policy_state.detach().clone()
            policy_next_state_raw = policy_next_state.detach().clone()
            expert_state_raw = expert_state.detach().clone()
            expert_next_state_raw = expert_next_state.detach().clone()

            b_pol = policy_state.size(0)
            disc_input = torch.cat(
                (
                    torch.cat([policy_state, policy_next_state], dim=-1),
                    torch.cat([expert_state, expert_next_state], dim=-1),
                ),
                dim=0,
            )
            disc_output = self.discriminator(disc_input)
            policy_d, expert_d = disc_output[:b_pol], disc_output[b_pol:]

            amp_loss, grad_pen_loss = self.discriminator.compute_loss(
                policy_d=policy_d,
                expert_d=expert_d,
                sample_amp_expert=(expert_state, expert_next_state),
                sample_amp_policy=(policy_state, policy_next_state),
                lambda_=10.0,
            )

            loss = ppo_loss + amp_loss + grad_pen_loss

            # --- Backward pass ---
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # --- Update discriminator normalizer ---
            self.discriminator.update_normalization(
                expert_state_raw,
                expert_next_state_raw,
                policy_state_raw,
                policy_next_state_raw,
            )

            # --- Accumulate statistics ---
            policy_d_prob = torch.sigmoid(policy_d)
            expert_d_prob = torch.sigmoid(expert_d)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d_prob.mean().item()
            mean_expert_pred += expert_d_prob.mean().item()
            mean_accuracy_policy += torch.sum(
                torch.round(policy_d_prob) == torch.zeros_like(policy_d_prob)
            ).item()
            mean_accuracy_expert += torch.sum(
                torch.round(expert_d_prob) == torch.ones_like(expert_d_prob)
            ).item()
            mean_accuracy_expert_elem += expert_d_prob.numel()
            mean_accuracy_policy_elem += policy_d_prob.numel()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_accuracy_policy /= max(1, mean_accuracy_policy_elem)
        mean_accuracy_expert /= max(1, mean_accuracy_expert_elem)
        mean_kl_divergence /= num_updates

        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            mean_accuracy_policy,
            mean_accuracy_expert,
            mean_kl_divergence,
        )

    def train_mode(self) -> None:
        """Set models to training mode."""
        self.actor.train()
        self.critic.train()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Set models to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.discriminator.eval()

    def get_policy(self) -> MLPModel:
        """Return the actor (policy) model."""
        return self.actor
