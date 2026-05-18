# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.amp_ppo import AmpPPO

__all__ = ["PPO", "Distillation", "AmpPPO"]
