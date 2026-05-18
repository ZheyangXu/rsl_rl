# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
from typing import Tuple

from rsl_rl.modules.normalization import EmpiricalNormalization


class Discriminator(nn.Module):
    """AMP discriminator network for distinguishing expert vs policy motion.

    Trained with LSGAN loss (AMP paper Section 5.2): target +1 for expert
    state transitions, -1 for policy transitions. Includes a gradient penalty
    (R1 regularizer, Section 5.4) on real data for training stability.

    A minibatch standard deviation feature is appended after the trunk to
    encourage diversity in policy-generated motions.

    Args:
        input_dim: Dimension of concatenated (state, next_state) input.
        hidden_layer_sizes: List of hidden layer dimensions.
        reward_scale: Scale factor applied to the style reward.
        loss_type: "LSGAN" (least-squares), "BCEWithLogits", or "Wasserstein".
        eta_wgan: Scaling factor for Wasserstein loss (if used).
        empirical_normalization: If True, normalize AMP observations before scoring.
        device: Torch device.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        loss_type: str = "LSGAN",
        eta_wgan: float = 0.3,
        empirical_normalization: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.reward_scale = reward_scale
        self.loss_type = loss_type

        # Build MLP trunk
        layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Linear head: input from trunk + 1 minibatch std feature
        final_in_dim = hidden_layer_sizes[-1] + 1
        self.linear = nn.Linear(final_in_dim, 1)

        # Empirical normalization (optional)
        self.empirical_normalization = empirical_normalization
        amp_obs_dim = input_dim // 2
        if empirical_normalization:
            self.amp_normalizer = EmpiricalNormalization(shape=[amp_obs_dim])
        else:
            self.amp_normalizer = nn.Identity()

        # Configure loss
        if loss_type == "Wasserstein":
            self.eta_wgan = eta_wgan
        self._loss_fn = {
            "LSGAN": None,
            "BCEWithLogits": torch.nn.BCEWithLogitsLoss(),
            "Wasserstein": None,
        }[loss_type]

        self.to(self.device)
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Splits the input into state and next_state, applies normalization,
        passes through trunk + minibatch-std → linear head.

        Args:
            x: Concatenated (state, next_state) tensor (B, input_dim).

        Returns:
            Discriminator logits (B, 1).
        """
        half = self.input_dim // 2
        state, next_state = torch.split(x, half, dim=-1)
        state = self.amp_normalizer(state)
        next_state = self.amp_normalizer(next_state)
        x_norm = torch.cat([state, next_state], dim=-1)

        h = self.trunk(x_norm)
        s = self._minibatch_std_scalar(h)
        h = torch.cat([h, s], dim=-1)
        return self.linear(h)

    @staticmethod
    def _minibatch_std_scalar(h: torch.Tensor) -> torch.Tensor:
        """Mean over feature-wise std across the batch (B, 1)."""
        if h.shape[0] <= 1:
            return h.new_zeros((h.shape[0], 1))
        s = h.float().std(dim=0, unbiased=False).mean()
        return s.expand(h.shape[0], 1).to(h.dtype)

    def predict_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the AMP style reward from discriminator scores.

        For LSGAN (Equation 7 in paper):
            r = max(0, 1 - 0.25 * (D - 1)^2)

        For BCE: r = -log(1 - sigmoid(D)) = softplus(D)
        For Wasserstein: r = exp(tanh(eta * D))

        Args:
            state: Current state tensor (B, obs_dim).
            next_state: Next state tensor (B, obs_dim).

        Returns:
            Style reward tensor (B,).
        """
        with torch.no_grad():
            logit = self.forward(torch.cat([state, next_state], dim=-1))

            if self.loss_type == "Wasserstein":
                logit = torch.tanh(self.eta_wgan * logit)
                reward = self.reward_scale * torch.exp(logit).squeeze(1)
            elif self.loss_type == "LSGAN":
                # r = max(0, 1 - 0.25 * (D - 1)^2)
                reward = 1 - 0.25 * (logit - 1) ** 2
                reward = torch.clamp(reward, min=0.0)
                reward = self.reward_scale * reward.squeeze(1)
            else:
                # BCE: softplus(logit) = -log(1 - sigmoid(logit))
                reward = self.reward_scale * F.softplus(logit).squeeze(1)
            return reward

    def compute_loss(
        self,
        policy_d: torch.Tensor,
        expert_d: torch.Tensor,
        sample_amp_expert: Tuple[torch.Tensor, torch.Tensor],
        sample_amp_policy: Tuple[torch.Tensor, torch.Tensor],
        lambda_: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the discriminator loss and gradient penalty.

        Args:
            policy_d: Discriminator output for policy data (B_pol, 1).
            expert_d: Discriminator output for expert data (B_exp, 1).
            sample_amp_expert: Tuple of (expert_state, expert_next_state).
            sample_amp_policy: Tuple of (policy_state, policy_next_state).
            lambda_: Gradient penalty coefficient (default 10, as in paper).

        Returns:
            Tuple of (amp_loss, grad_pen_loss).
        """
        # Normalize for gradient penalty
        expert_state_n, expert_next_n = (
            self.amp_normalizer(sample_amp_expert[0]),
            self.amp_normalizer(sample_amp_expert[1]),
        )
        policy_state_n, policy_next_n = (
            self.amp_normalizer(sample_amp_policy[0]),
            self.amp_normalizer(sample_amp_policy[1]),
        )

        grad_pen_loss = self._compute_grad_pen(
            expert_states=(expert_state_n, expert_next_n),
            policy_states=(policy_state_n, policy_next_n),
            lambda_=lambda_,
        )

        if self.loss_type == "LSGAN":
            expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
        elif self.loss_type == "BCEWithLogits":
            expert_loss = self._loss_fn(expert_d, torch.ones_like(expert_d))
            policy_loss = self._loss_fn(policy_d, torch.zeros_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
        elif self.loss_type == "Wasserstein":
            amp_loss = self._wgan_loss(policy_d=policy_d, expert_d=expert_d)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return amp_loss, grad_pen_loss

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """Update empirical normalizer statistics with new observations.

        Args:
            *batches: One or more batches of (unnormalized) observations.
        """
        if not self.empirical_normalization:
            return
        with torch.no_grad():
            for batch in batches:
                self.amp_normalizer.update(batch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_grad_pen(
        self,
        expert_states: Tuple[torch.Tensor, torch.Tensor],
        policy_states: Tuple[torch.Tensor, torch.Tensor],
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """R1 gradient penalty on real (expert) data."""
        expert = torch.cat(expert_states, -1)

        if self.loss_type == "Wasserstein":
            policy = torch.cat(policy_states, -1)
            alpha = torch.rand(expert.size(0), 1, device=expert.device)
            alpha = alpha.expand_as(expert)
            data = alpha * expert + (1 - alpha) * policy
            data = data.detach().requires_grad_(True)
            h = self.trunk(data)
            s = self._minibatch_std_scalar(h)
            h_cat = torch.cat([h, s], dim=-1)
            scores = self.linear(h_cat)
            grad = autograd.grad(
                outputs=scores,
                inputs=data,
                grad_outputs=torch.ones_like(scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return lambda_ * (grad.norm(2, dim=1) - 1.0).pow(2).mean()

        # R1 regularizer: 0.5 * lambda * ||grad_x D(x_real)||^2
        data = expert.detach().requires_grad_(True)
        h = self.trunk(data)
        with torch.no_grad():
            s = self._minibatch_std_scalar(h)
        h_cat = torch.cat([h, s], dim=-1)
        scores = self.linear(h_cat)

        grad = autograd.grad(
            outputs=scores.sum(),
            inputs=data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return 0.5 * lambda_ * (grad.pow(2).sum(dim=1)).mean()

    def _wgan_loss(
        self,
        policy_d: torch.Tensor,
        expert_d: torch.Tensor,
    ) -> torch.Tensor:
        """Modified Wasserstein loss with tanh stabilization."""
        policy_d = torch.tanh(self.eta_wgan * policy_d)
        expert_d = torch.tanh(self.eta_wgan * expert_d)
        return policy_d.mean() - expert_d.mean()
