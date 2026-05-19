# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Generator, Tuple

import torch


class ReplayBuffer:
    """Fixed-size circular buffer for (state, next_state) transition pairs.

    Used by the AMP algorithm to prevent the discriminator from overfitting
    to the most recent policy trajectories (AMP paper Section 6.3).

    Attributes:
        states: Buffer of current states (buffer_size, obs_dim).
        next_states: Buffer of next states (buffer_size, obs_dim).
        buffer_size: Maximum number of transitions to store.
        device: Torch device for tensor storage.
        step: Current write index (circular).
        num_samples: Total number of valid samples in the buffer.
    """

    def __init__(
        self,
        obs_dim: int,
        buffer_size: int,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the replay buffer.

        Args:
            obs_dim: Dimension of the observation space.
            buffer_size: Maximum number of transitions to store.
            device: Torch device for buffer allocation.
        """
        self.device = torch.device(device)
        self.buffer_size = buffer_size

        self.states = torch.zeros(
            (buffer_size, obs_dim), dtype=torch.float32, device=self.device
        )
        self.next_states = torch.zeros(
            (buffer_size, obs_dim), dtype=torch.float32, device=self.device
        )

        self.step = 0
        self.num_samples = 0

    def insert(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> None:
        """Insert a batch of (state, next_state) pairs into the buffer.

        Args:
            states: Batch of current states (batch_size, obs_dim).
            next_states: Batch of next states (batch_size, obs_dim).
        """
        states = states.to(self.device)
        next_states = next_states.to(self.device)

        batch_size = states.shape[0]

        # If inserting more than the buffer can hold, only keep the most recent
        if batch_size > self.buffer_size:
            self.states[:] = states[-self.buffer_size:]
            self.next_states[:] = next_states[-self.buffer_size:]
            self.step = 0
            self.num_samples = self.buffer_size
            return

        end = self.step + batch_size

        if end <= self.buffer_size:
            self.states[self.step:end] = states
            self.next_states[self.step:end] = next_states
        else:
            first_part = self.buffer_size - self.step
            self.states[self.step:] = states[:first_part]
            self.next_states[self.step:] = next_states[:first_part]
            remainder = batch_size - first_part
            self.states[:remainder] = states[first_part:]
            self.next_states[:remainder] = next_states[first_part:]

        self.step = end % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + batch_size)

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        allow_replacement: bool = True,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yield mini-batches of (state, next_state) tuples from the buffer.

        Args:
            num_mini_batch: Number of mini-batches to yield.
            mini_batch_size: Number of samples per mini-batch.
            allow_replacement: If True, sample with replacement when the
                request exceeds stored samples. Defaults to True.

        Yields:
            Tuple of (state_batch, next_state_batch) tensors.

        Raises:
            ValueError: If the total request exceeds stored samples and
                allow_replacement is False.
        """
        total = num_mini_batch * mini_batch_size

        if total > self.num_samples:
            if not allow_replacement:
                raise ValueError(
                    f"Not enough samples in buffer: requested {total}, "
                    f"but have {self.num_samples}"
                )
            cycles = (total + self.num_samples - 1) // self.num_samples
            big_size = self.num_samples * cycles
            big_perm = torch.randperm(big_size, device=self.device)
            indices = big_perm[:total] % self.num_samples
        else:
            indices = torch.randperm(self.num_samples, device=self.device)[:total]

        for i in range(num_mini_batch):
            batch_idx = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
            yield self.states[batch_idx], self.next_states[batch_idx]

    def __len__(self) -> int:
        """Return the number of valid samples currently stored."""
        return self.num_samples
