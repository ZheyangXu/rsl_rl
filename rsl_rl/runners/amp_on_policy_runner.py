# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.amp_ppo import AmpPPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import check_nan, resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger
from rsl_rl.utils.motion_loader import AmpLoader


class AmpOnPolicyRunner:
    """On-policy runner for AMP (Adversarial Motion Priors) + PPO training.

    Orchestrates the training loop: rollout collection with combined
    task+style rewards, discriminator-based style reward computation,
    and the interleaved PPO+discriminator update.

    The runner expects an AMP-compatible environment that provides
    ``observations["amp"]`` containing the AMP observation group.

    Args:
        env: The vectorized environment.
        train_cfg: Training configuration dictionary.
        log_dir: Directory for logging and checkpoints.
        device: Torch device.
    """

    alg: AmpPPO
    """The AMP+PPO algorithm instance."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Resolve observation groups (policy, critic)
        obs = self.env.get_observations()
        default_sets = ["actor", "critic"]
        self.cfg["obs_groups"] = resolve_obs_groups(
            obs, self.cfg.get("obs_groups"), default_sets
        )

        # Determine the AMP observation key from the environment
        # The discriminator observation group in the env is typically named "disc"
        if "disc" in obs:
            self._amp_obs_key = "disc"
        elif "amp" in obs:
            self._amp_obs_key = "amp"
        else:
            raise KeyError(
                "AMP observations not found in env. Expected 'disc' or 'amp' key "
                f"in observation TensorDict, got keys: {list(obs.keys())}"
            )

        # --- Build actor and critic ---
        actor_class: type[MLPModel] = resolve_callable(self.cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(self.cfg["critic"].pop("class_name"))

        actor = actor_class(
            obs, self.cfg["obs_groups"], "actor",
            self.env.num_actions, **self.cfg["actor"],
        ).to(self.device)
        critic = critic_class(
            obs, self.cfg["obs_groups"], "critic",
            1, **self.cfg["critic"],
        ).to(self.device)
        print(f"Actor Model: {actor}")
        print(f"Critic Model: {critic}")

        # --- Build AMP data loader and discriminator ---
        amp_cfg = self.cfg["algorithm"]["amp_cfg"]
        dataset_cfg = amp_cfg.get("dataset", {})
        disc_cfg = amp_cfg["amp_discriminator"]

        num_amp_obs = self._flatten_amp_obs(obs[self._amp_obs_key]).shape[-1]

        # Resolve AMP joint names from environment if not provided
        amp_joint_names = dataset_cfg.get("amp_joint_names", None)
        if amp_joint_names is None:
            try:
                params = self.env.cfg.observations.disc.joint_pos.params
                if params and "asset_cfg" in params:
                    amp_joint_names = params["asset_cfg"].joint_names
            except (AttributeError, KeyError):
                amp_joint_names = None

        sim_cfg = getattr(self.env.cfg, "sim", None)
        if sim_cfg is None or not hasattr(sim_cfg, "dt"):
            raise AttributeError(
                "env.cfg.sim.dt is required. Ensure your environment config defines sim.dt."
            )
        if not hasattr(self.env.cfg, "decimation"):
            raise AttributeError(
                "env.cfg.decimation is required. Ensure your environment config defines decimation."
            )
        simulation_dt = self.env.cfg.sim.dt * self.env.cfg.decimation

        amp_data = AmpLoader(
            device=self.device,
            dataset_path_root=dataset_cfg["amp_data_path"],
            datasets=dataset_cfg["datasets"],
            simulation_dt=simulation_dt,
            slow_down_factor=dataset_cfg.get("slow_down_factor", 1),
            expected_joint_names=amp_joint_names,
        )

        self.discriminator = Discriminator(
            input_dim=num_amp_obs * 2,
            hidden_layer_sizes=disc_cfg["hidden_dims"],
            reward_scale=disc_cfg.get("style_reward_scale", 1.0),
            loss_type=amp_cfg.get("loss_type", "LSGAN"),
            empirical_normalization=amp_cfg.get("empirical_normalization", False),
            device=self.device,
        ).to(self.device)

        # --- Build algorithm ---
        alg_class: type[AmpPPO] = resolve_callable(self.cfg["algorithm"].pop("class_name"))

        self.style_weight = amp_cfg.get("task_style_lerp", 0.5)

        # Filter algorithm kwargs to match AmpPPO.__init__ signature
        alg_kwargs = {}
        valid_keys = set(AmpPPO.__init__.__code__.co_varnames)
        storage = RolloutStorage(
            "rl", env.num_envs, self.cfg["num_steps_per_env"],
            obs, [env.num_actions], device,
        )
        alg_kwargs["storage"] = storage
        alg_kwargs["amp_replay_buffer_size"] = amp_cfg.get("disc_obs_buffer_size", 100000)

        for key, value in self.cfg["algorithm"].items():
            if key in valid_keys:
                alg_kwargs[key] = value

        self.alg: AmpPPO = alg_class(
            actor=actor,
            critic=critic,
            discriminator=self.discriminator,
            amp_data=amp_data,
            device=self.device,
            **alg_kwargs,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # --- Logger ---
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=False,
            gpu_world_size=1,
            gpu_global_rank=0,
            device=self.device,
        )
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the AMP+PPO learning loop.

        Args:
            num_learning_iterations: Number of iterations to train.
            init_at_random_ep_len: If True, randomize initial episode lengths.
        """
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        amp_obs = self._flatten_amp_obs(obs[self._amp_obs_key]).clone()
        self.alg.train_mode()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            start = time.time()

            # --- Rollout ---
            mean_task_reward_log = 0.0
            mean_style_reward_log = 0.0

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    self.alg.act_amp(amp_obs)

                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    next_amp_obs = self._flatten_amp_obs(obs[self._amp_obs_key]).clone()
                    style_rewards = self.discriminator.predict_reward(
                        amp_obs, next_amp_obs
                    )

                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()

                    # Combined reward: (1 - w) * task + w * style
                    rewards = (
                        (1 - self.style_weight) * rewards
                        + self.style_weight * style_rewards
                    )

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.alg.process_amp_step(next_amp_obs)

                    amp_obs = next_amp_obs

                    # Book keeping
                    intrinsic_rewards = None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # --- Update ---
            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, \
                mean_policy_pred, mean_expert_pred, mean_accuracy_policy, \
                mean_accuracy_expert, mean_kl_divergence = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env

            # --- Logging ---
            loss_dict = {
                "value": mean_value_loss,
                "surrogate": mean_surrogate_loss,
                "amp": mean_amp_loss,
                "amp_grad_pen": mean_grad_pen_loss,
                "amp_policy_pred": mean_policy_pred,
                "amp_expert_pred": mean_expert_pred,
                "amp_accuracy_policy": mean_accuracy_policy,
                "amp_accuracy_expert": mean_accuracy_expert,
            }
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            # --- Save ---
            if self.logger.writer is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        # Final save
        if self.logger.writer is not None:
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
            )
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save models and training state."""
        saved_dict = {
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load models and training state."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        if load_cfg is None:
            load_cfg = {}

        if load_cfg.get("actor", True) and "actor_state_dict" in loaded_dict:
            self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic", True) and "critic_state_dict" in loaded_dict:
            self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("discriminator", True) and "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"], strict=False
            )
        if load_cfg.get("optimizer", True) and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]

        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy for inference."""
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    @staticmethod
    def _flatten_amp_obs(amp_obs: TensorDict | dict | torch.Tensor) -> torch.Tensor:
        """Flatten AMP observations into a single 2D tensor (B, dim).

        Handles TensorDict, dict, and plain tensor inputs.
        """
        if isinstance(amp_obs, torch.Tensor):
            return amp_obs
        # Try common AMP observation keys
        if isinstance(amp_obs, (dict, TensorDict)):
            if "joint_pos" in amp_obs and "joint_vel" in amp_obs:
                return torch.cat([amp_obs["joint_pos"], amp_obs["joint_vel"]], dim=-1)
            keys = sorted(amp_obs.keys())
            return torch.cat([amp_obs[k] for k in keys], dim=-1)
        if hasattr(amp_obs, "keys"):
            return AmpOnPolicyRunner._flatten_amp_obs(dict(amp_obs.items()))
        raise TypeError(f"Unsupported AMP observation type: {type(amp_obs)}")
