# AMP (Adversarial Motion Priors) Implementation Plan

> Reference Paper: [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control](https://doi.org/10.1145/3450626.3459670)
> Authors: Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa (2021)

## Overview

This plan covers two main work streams:

1. **Part 1**: Add AMP algorithm support to `rsl_rl` (v5.3) under `/opt/zouyu-workspaces/rsl_rl/`
2. **Part 2**: Add Unitree G1 AMP task to `zouyu/luwu` under `/opt/zouyu-workspaces/zouyu/luwu/`

Reference projects:
- `/opt/zouyu-workspaces/amp/amp-rsl-rl/` — AMP implementation for rsl_rl
- `/opt/zouyu-workspaces/amp/legged_lab/` — AMP environment/task config for IsaacLab/legged_lab
- `/opt/zouyu-workspaces/TienKung-Lab/` — Alternative AMP reference with TienKung modifications

### Clarifications

- **G1 motion data**: Located at `/opt/zouyu-workspaces/amp/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run` (`.pkl` format, not `.npy`).
- **No legged_lab dependency**: We do NOT include the legged_lab-style `ManagerBasedAmpEnvCfg` base class. The luwu package already has its own environment infrastructure via IsaacLab's `ManagerBasedRLEnvCfg`. AMP env configs will directly inherit from IsaacLab base classes.
- **rsl_rl 5.3 only**: No backward compatibility shims. Clean implementation targeting the current rsl_rl v5.3 API (TensorDict observations, separate actor/critic models, new RolloutStorage API).

### Coding Conventions

- **Import style**: Always use fully-qualified import paths (e.g., `from rsl_rl.storage.replay_buffer import ReplayBuffer`), not relative imports (e.g., `from .replay_buffer import ...`).
- **Naming**: Python class names use PascalCase (`AmpPPO`, `AmpLoader`). Variable names, function names, and file names use snake_case (`amp_ppo.py`, `amp_loader.py`, `style_weight`).
- **No ALL_CAPS abbreviations in class names**: `AmpPPO` instead of `AMP_PPO`, `AmpLoader` instead of `AMPLoader`, `AmpOnPolicyRunner` instead of `AMPOnPolicyRunner`.

---

## Part 1: rsl_rl Core Changes

### 1.1 New File: `rsl_rl/rsl_rl/storage/replay_buffer.py`

Fixed-size circular buffer storing `(state, next_state)` transition pairs for discriminator training.

**Why**: The AMP paper (Section 6.3) uses a replay buffer to prevent the discriminator from overfitting to the most recent policy trajectories.

```python
class ReplayBuffer:
    """Fixed-size circular buffer for (state, next_state) pairs."""
    def __init__(self, obs_dim: int, buffer_size: int, device: str = "cpu")
    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None
    def feed_forward_generator(self, num_mini_batch, mini_batch_size, allow_replacement=True)
    def __len__(self) -> int
```

Key details:
- Pre-allocated tensors on target device
- Circular insertion with wrap-around
- Generator supports sampling with/without replacement
- Based on `amp-rsl-rl/amp_rsl_rl/storage/replay_buffer.py`

### 1.2 New File: `rsl_rl/rsl_rl/modules/discriminator.py`

The AMP discriminator network that distinguishes expert motion from policy-generated motion.

**Why**: Core component of the AMP algorithm — acts as the "motion prior" that scores how similar a state transition is to the reference dataset.

```python
class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim,               # 2 * amp_obs_dim (concat of s_t and s_{t+1})
        hidden_layer_sizes,      # e.g., [1024, 512]
        reward_scale,            # scale factor for style reward
        loss_type="LSGAN",       # "LSGAN" | "BCEWithLogits" | "Wasserstein"
        empirical_normalization=False,
        device="cpu",
    )
```

**Architecture**:
- MLP trunk: `Linear → ReLU → ... → Linear → ReLU`
- Minibatch standard deviation feature appended after trunk (stabilizes GAN training)
- Linear output layer → scalar score

**LSGAN Loss** (Equation 6 in paper):
- `loss = E[(D(expert) - 1)^2] + E[(D(policy) + 1)^2]`
- Target: +1 for expert transitions, -1 for policy transitions

**Style Reward** (Equation 7 in paper):
- `r_S = max(0, 1 - 0.25 * (D - 1)^2)`
- Bounded in [0, 1], encourages policy to produce transitions that score +1

**Gradient Penalty (R1)** (Equation 8 in paper):
- `0.5 * lambda * ||grad_x D(x_real)||^2`
- Applied only to real (expert) data
- `lambda = 10` (as used in paper)

**Key Methods**:
| Method | Description |
|--------|-------------|
| `forward(x)` | Normalizes input (if enabled), passes through trunk + linear head |
| `predict_reward(state, next_state)` | Returns style reward from discriminator scores |
| `compute_loss(policy_d, expert_d, ...)` | Computes amp_loss + grad_pen_loss |
| `compute_grad_pen(expert_states, policy_states, lambda_)` | R1 gradient penalty |
| `update_normalization(*batches)` | Updates empirical normalizer statistics |

**Discriminator Observations** (Section 5.3):
- Root linear velocity & angular velocity (local frame)
- Local rotation of each joint (6D tangent-normal encoding)
- Local velocity of each joint
- 3D positions of end-effectors (local frame)

### 1.3 New File: `rsl_rl/rsl_rl/algorithms/amp_ppo.py`

The `AmpPPO` class extending the PPO algorithm with discriminator-based style reward.

**Why**: This is the main algorithm — it combines PPO for policy optimization with adversarial training of the discriminator.

```python
class AmpPPO:
    def __init__(
        self,
        actor, critic, discriminator, amp_data,
        num_learning_epochs, num_mini_batches,
        clip_param, gamma, lam,
        value_loss_coef, entropy_coef,
        learning_rate, max_grad_norm,
        use_clipped_value_loss, schedule, desired_kl,
        amp_replay_buffer_size=100000,
        device="cpu",
    )
```

**Architecture (vs base PPO)**:
- Follows `rsl_rl.algorithms.PPO` pattern (TensorDict observations, separate actor/critic models)
- Adds: `discriminator`, `amp_data` (AmpLoader), `amp_storage` (ReplayBuffer)
- Separate parameter groups with weight decay for discriminator trunk (1e-4) and head (1e-2)

**Key Methods**:
| Method | Description |
|--------|-------------|
| `act(obs)` | Sample actions + store transition (same as base PPO) |
| `act_amp(amp_obs)` | Store current AMP observation for replay buffer |
| `process_env_step(obs, rewards, dones, extras)` | Record env step + update normalizers |
| `process_amp_step(next_amp_obs)` | Insert (prev_obs, next_obs) into replay buffer |
| `compute_returns(obs)` | GAE-lambda returns (same as base PPO) |
| `update()` | PPO update + discriminator update (interleaved) |

**Update Loop** (per mini-batch):
1. Unpack PPO batch → compute surrogate loss + value loss + entropy
2. Sample policy transitions from replay buffer
3. Sample expert transitions from AmpLoader
4. Discriminator forward on both → compute amp_loss + grad_pen_loss
5. Total loss: `ppo_loss + amp_loss + grad_pen_loss`
6. Backward pass, gradient clipping, optimizer step
7. Update discriminator normalizer statistics

**Returns**: Tuple of 9 mean metrics:
`(value_loss, surrogate_loss, amp_loss, grad_pen_loss, policy_pred, expert_pred, accuracy_policy, accuracy_expert, kl_divergence)`

### 1.4 New File: `rsl_rl/rsl_rl/utils/motion_loader.py`

Motion data loader that processes mocap `.npy` (or `.pkl`) files into AMP-compatible format.

**Why**: The reference motion data needs to be loaded, resampled to match the simulator timestep, and converted to the correct coordinate conventions.

```python
@dataclass
class MotionData:
    """Stores processed motion data for one clip."""
    joint_positions       # (T, N)
    joint_velocities      # (T, N)
    base_lin_velocities_mixed   # (T, 3) world frame
    base_ang_velocities_mixed   # (T, 3) world frame
    base_lin_velocities_local   # (T, 3) body frame
    base_ang_velocities_local   # (T, 3) body frame
    base_quat             # (T, 4) wxyz format

class AmpLoader:
    def __init__(
        self,
        device, dataset_path_root, datasets,  # datasets = {"name": weight, ...}
        simulation_dt, slow_down_factor,
        expected_joint_names=None,
    )
```

**Dataset Format** (`.npy` or `.pkl` files):
```python
{
    "joints_list": List[str],           # ordered joint names
    "joint_positions": List[np.ndarray], # per-frame joint configurations
    "root_position": List[np.ndarray],   # base position (world)
    "root_quaternion": List[np.ndarray], # base orientation (xyzw, SciPy convention)
    "fps": float,                        # original framerate
}
```

**Processing Pipeline**:
1. Load data files (supports both `.npy` and `.pkl`), build joint union ordering
2. SLERP-interpolate quaternions to match `simulation_dt`
3. Finite-difference to compute joint velocities
4. Convert base lin/ang velocities to local frame
5. Convert quaternions from `xyzw` → `wxyz` (IsaacLab convention)
6. Build per-frame sampling weights for balanced training

**Key Methods**:
| Method | Description |
|--------|-------------|
| `feed_forward_generator(n_batches, batch_size)` | Yields (state, next_state) pairs from expert data |
| `get_state_for_reset(n_samples)` | Returns full state for reference state initialization |

### 1.5 New File: `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py`

The `AmpOnPolicyRunner` orchestrating the complete AMP training loop.

**Why**: The runner handles the reward mixing (task + style), AMP observation collection, and discriminator-based style reward computation during rollout.

```python
class AmpOnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu")
```

**Training Loop** (`learn()`):
```
for iteration in range(num_learning_iterations):
    # 1. Rollout
    for step in range(num_steps_per_env):
        actions = alg.act(obs)
        alg.act_amp(amp_obs)
        obs, task_rewards, dones, extras = env.step(actions)

        next_amp_obs = flatten(obs["amp"])
        style_rewards = discriminator.predict_reward(amp_obs, next_amp_obs)

        # Combined reward (Equation 4)
        rewards = (1 - style_weight) * task_rewards + style_weight * style_rewards

        alg.process_env_step(obs, rewards, dones, extras)
        alg.process_amp_step(next_amp_obs)

    # 2. Compute returns
    alg.compute_returns(obs)

    # 3. Update (PPO + discriminator)
    metrics = alg.update()

    # 4. Log & save
```

**Reward Mixing**:
- `style_weight` controls the tradeoff between task and style objectives
- Paper uses `w_G = 0.5, w_S = 0.5` (Section 8.2)
- Configurable via `train_cfg["style_weight"]`

**AMP Observation Handling**:
- The environment provides AMP observations via `obs["amp"]` (TensorDict key)
- `_flatten_amp_obs()` converts dict/TensorDict to flat tensor for discriminator

**Checkpointing**:
- Saves: actor, critic, discriminator, optimizer state, iteration
- Optional ONNX export of policy

### 1.6 Updated `__init__.py` Files

**`rsl_rl/rsl_rl/storage/__init__.py`**:
```python
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer

__all__ = ["RolloutStorage", "ReplayBuffer"]
```

**`rsl_rl/rsl_rl/modules/__init__.py`**:
```python
from rsl_rl.modules.discriminator import Discriminator
# ... existing imports ...

__all__ = [..., "Discriminator"]
```

**`rsl_rl/rsl_rl/algorithms/__init__.py`**:
```python
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.algorithms.amp_ppo import AmpPPO

__all__ = ["PPO", "Distillation", "AmpPPO"]
```

**`rsl_rl/rsl_rl/utils/__init__.py`**:
```python
from rsl_rl.utils.motion_loader import AmpLoader, MotionData
# ... existing imports ...

__all__ = [..., "AmpLoader", "MotionData"]
```

**`rsl_rl/rsl_rl/runners/__init__.py`**:
```python
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.runners.distillation_runner import DistillationRunner
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner

__all__ = ["DistillationRunner", "OnPolicyRunner", "AmpOnPolicyRunner"]
```

---

## Part 2: zouyu/luwu Task Changes

### 2.1 New File: `zouyu/luwu/source/luwu/luwu/assets/robots/unitree_g1.py`

Unitree G1 29-DOF humanoid robot configuration.

Based on `amp/legged_lab/source/legged_lab/legged_lab/assets/unitree.py`.

```python
UNITREE_G1_29DOF_CFG = ArticulationCfg(
    spawn=UsdFileCfg(usd_path="..."),
    init_state=InitialStateCfg(pos=(0, 0, 0.8), joint_pos={...}),
    actuators={...},  # 4 actuator groups with different specs
    joint_sdk_names=[...],  # 29 joints
)
```

**29 Joints**: hip_pitch/roll/yaw, knee, ankle_pitch/roll (×2), waist_yaw/roll/pitch,
shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw (×2)

### 2.2 New Directory: `zouyu/luwu/source/luwu/luwu/tasks/locomotion/amp/`

The AMP task directly inherits from IsaacLab's `ManagerBasedRLEnvCfg` (no legged_lab middleware).

#### 2.2.1 `amp/__init__.py`
Module init, exports the base AMP task configs.

#### 2.2.2 `amp/amp_env_cfg.py` — Base AMP Locomotion Environment

```python
from isaaclab.envs import ManagerBasedRLEnvCfg

@configclass
class AmpSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg     # flat plane terrain
    robot: ArticulationCfg          # MISSING (set by subclass)
    robot_anim: ArticulationCfg     # reference animation robot (optional)
    contact_forces: ContactSensorCfg
    sky_light: DomeLightCfg

@configclass
class CommandsCfg:
    base_velocity: UniformVelocityCommandCfg(
        heading_command=True,
        ranges=(lin_vel_x, lin_vel_y, ang_vel_z, heading)
    )

@configclass
class ActionsCfg:
    joint_pos: JointPositionActionCfg  # PD target positions

@configclass
class ObservationsCfg:
    # 4 observation groups:
    policy: PolicyCfg               # agent observations (history=5)
    critic: CriticCfg               # privileged observations
    disc: DiscriminatorCfg          # AMP discriminator input (history=4, no flatten)
    disc_demo: DiscriminatorDemoCfg # Reference motion for discriminator (no history)

@configclass
class LocomotionAmpEnvCfg(ManagerBasedRLEnvCfg):
    scene: AmpSceneCfg
    observations: ObservationsCfg
    actions: ActionsCfg
    commands: CommandsCfg
    rewards: RewardsCfg
    terminations: TerminationsCfg
    events: EventCfg
    curriculum: CurriculumCfg
    motion_data: MotionDataCfg
    animation: AnimationCfg
    # defaults: decimation=4, episode_length_s=20.0, sim.dt=0.005
```

Key difference from legged_lab version: directly uses `ManagerBasedRLEnvCfg` as base class, with motion_data and animation as extra config attributes.

#### 2.2.3 `amp/mdp/` — MDP Functions

**Observations** (`mdp/observations.py`):
| Function | Description |
|----------|-------------|
| `root_local_rot_tan_norm` | 6D tangent-normal rotation encoding of root |
| `ref_root_local_rot_tan_norm` | Same for reference motion |
| `ref_root_ang_vel_b` | Reference root angular velocity (body frame) |
| `ref_joint_pos` / `ref_joint_vel` | Reference joint positions/velocities |
| `ref_key_body_pos_b` | Reference end-effector positions (body frame) |
| `key_body_pos_b` | Current end-effector positions (body frame) |

**Rewards** (`mdp/rewards.py`):
| Function | Weight | Description |
|----------|--------|-------------|
| `track_lin_vel_xy_exp` | 1.0 | Track XY linear velocity (exponential) |
| `track_ang_vel_z_exp` | 0.5-1.0 | Track Z angular velocity |
| `flat_orientation_l2` | -1.0 | Penalize non-flat base orientation |
| `lin_vel_z_l2` | -0.2 | Penalize Z linear velocity |
| `ang_vel_xy_l2` | -0.05 | Penalize XY angular velocity |
| `dof_torques_l2` | -2e-6 | Penalize joint torques |
| `dof_acc_l2` | -1e-7 | Penalize joint accelerations |
| `action_rate_l2` | -0.005 | Penalize action rate of change |
| `dof_pos_limits` | -1.0 | Penalize joint position limits |
| `joint_deviation_l1` | -0.05~-0.1 | Penalize deviation from default pose |
| `feet_air_time_positive_biped` | 0.5 | Encourage foot air time (biped gait) |
| `feet_slide` | -0.1 | Penalize foot sliding |
| `is_terminated` | -200.0 | Penalize early termination |
| `undesired_contacts` | -1.0 | Penalize undesired body contacts |

**Terminations** (`mdp/terminations.py`):
- `time_out` — episode time limit
- `root_height_below_minimum` — root height below 0.2m
- `bad_orientation` — roll/pitch exceeding 60°
- `illegal_contact` — non-foot body contact with ground

**Events** (`mdp/events.py`):
- `reset_from_ref` — reset to random reference motion state (reference state initialization per AMP paper Section 6.3)
- `randomize_rigid_body_material` — randomize friction
- `randomize_rigid_body_mass` — randomize base mass
- `push_by_setting_velocity` — random pushes at intervals
- `apply_external_force_torque` — random force/torque on reset

### 2.3 New Directory: `zouyu/luwu/source/luwu/luwu/tasks/locomotion/amp/config/g1/`

#### 2.3.1 `g1/__init__.py`
Module init.

#### 2.3.2 `g1/g1_amp_env_cfg.py` — G1-Specific Config

```python
# Key body names for end-effector position observations
KEY_BODY_NAMES = [
    "left_ankle_roll_link",  "right_ankle_roll_link",
    "left_wrist_yaw_link",   "right_wrist_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4

@configclass
class G1AmpEnvCfg(LocomotionAmpEnvCfg):
    # Robot
    scene.robot = UNITREE_G1_29DOF_CFG

    # Motion dataset (walk_and_run .pkl files)
    motion_data.motion_dataset.motion_data_dir = (
        "/opt/zouyu-workspaces/amp/legged_lab/source/legged_lab/legged_lab/"
        "data/MotionData/g1_29dof/amp/walk_and_run"
    )
    motion_data.motion_dataset.motion_data_weights = {
        "Walk_*": 1.0, "run_*": 1.0, ...  # ~30 clips
    }

    # Observations: set key body params, history lengths
    observations.policy.key_body_pos_b.params = SceneEntityCfg("robot", body_names=KEY_BODY_NAMES)
    observations.critic.key_body_pos_b.params = ...
    observations.disc.key_body_pos_b.params = ...
    observations.disc.history_length = AMP_NUM_STEPS
    observations.disc_demo.ref_*_params["animation"] = ANIMATION_TERM_NAME

    # Events: body-specific settings
    events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
    events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
    events.reset_from_ref.params = {"animation": ANIMATION_TERM_NAME, "height_offset": 0.1}

    # Commands:
    #   lin_vel_x: (-0.5, 3.0)
    #   lin_vel_y: (-0.5, 0.5)
    #   ang_vel_z: (-1.0, 1.0)
    #   heading: (-pi, pi)

    # Terminations: disable base_contact termination (G1 has more body parts)

    # Animation: num_steps_to_use = AMP_NUM_STEPS

@configclass
class G1AmpEnvCfgPlay(G1AmpEnvCfg):
    # 48 envs, fixed heading=0, no reset_from_ref
```

#### 2.3.3 `g1/agents/__init__.py`
Module init.

#### 2.3.4 `g1/agents/rsl_rl_ppo_cfg.py` — Training Config

```python
@configclass
class G1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AmpOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "g1_amp"
    style_weight = 0.4  # task_style_lerp (1.0 - task_style_lerp = 0.6 task weight)

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }

    actor = RslRlPpoActorCriticCfg(
        class_name="MLPModel",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="AmpPPO",
        learning_rate=1e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            amp_discriminator=AmpDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.4,
            ),
            loss_type="LSGAN",
        ),
    )
```

### 2.4 New File: `zouyu/luwu/scripts/rsl_rl/train_amp.py`

New training script for AMP (similar to existing `train.py` but uses `AmpOnPolicyRunner`).

```python
"""Script to train RL agent with RSL-RL + AMP."""

# Key differences from train.py:
# - Import AmpOnPolicyRunner from rsl_rl.runners
# - Support AMP-specific config fields (style_weight, obs_groups, amp_cfg, etc.)
# - Register "AmpOnPolicyRunner" as a recognized runner class_name

# ...

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# ...

if agent_cfg.class_name == "AmpOnPolicyRunner":
    runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
elif agent_cfg.class_name == "OnPolicyRunner":
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
# ...
```

Note: Alternatively, we can simply modify the existing `train.py` to also handle `"AmpOnPolicyRunner"` as a runner class_name, avoiding code duplication. The script would check `agent_cfg.class_name` and instantiate the appropriate runner.

### 2.5 Updated `__init__.py` Files

- `luwu/tasks/locomotion/__init__.py`: Add `from luwu.tasks.locomotion.amp import *`
- `luwu/assets/robots/__init__.py`: Add `from luwu.assets.robots.unitree_g1 import UNITREE_G1_29DOF_CFG`

### 2.6 IsaacLab Task Registration

Register the G1 AMP task so it can be discovered by `gym.make()` and Hydra:

```
# In luwu/tasks/locomotion/amp/config/g1/__init__.py or a dedicated registration file:
# Register: "Luwu-Unitree-G1-Amp" -> G1AmpEnvCfg
# Register: "Luwu-Unitree-G1-Amp-Play" -> G1AmpEnvCfgPlay
# Register agent: "rsl_rl_cfg_entry_point" -> G1RslRlOnPolicyRunnerAmpCfg
```

---

## Algorithm Details (from Paper)

### Reward Formulation (Equation 4)
```
r(s_t, a_t, s_{t+1}, g) = w_G * r_G(task) + w_S * r_S(style)
```
- `r_G`: Task reward (e.g., velocity tracking) — manually designed
- `r_S`: Style reward — learned by discriminator
- Typical: `w_G = 0.6, w_S = 0.4` (i.e., `style_weight = 0.4`, `task_style_lerp = 0.4`)

### Discriminator (Section 5.1-5.4)
- **Input**: State transitions `(s_t, s_{t+1})` (not state-action pairs)
- **Architecture**: MLP with minibatch std feature
- **Loss**: Least-Squares GAN (LSGAN)
  - Expert target: +1
  - Policy target: -1
- **Gradient Penalty**: R1 regularizer on real data, λ=10
- **Observations**: root velocity, joint rotations/velocities, end-effector positions (local frame)

### Style Reward (Equation 7)
```
r_S(s_t, s_{t+1}) = max(0, 1 - 0.25 * (D(s_t, s_{t+1}) - 1)^2)
```
- Bounded in [0, 1]
- When D ≈ 1 (expert-like), reward ≈ 1
- When D ≈ -1 (policy-like), reward ≈ 0

### Training Details (Section 6.3)
- PPO with GAE-λ
- Reference state initialization (random reference motion states)
- Early termination (body-ground contact, except feet)
- Replay buffer for discriminator (prevents overfitting)
- Joint training of policy + discriminator (no pre-training needed)

---

## Change Summary

| # | File | Action |
|---|------|--------|
| 1 | `rsl_rl/rsl_rl/storage/replay_buffer.py` | **New** |
| 2 | `rsl_rl/rsl_rl/modules/discriminator.py` | **New** |
| 3 | `rsl_rl/rsl_rl/algorithms/amp_ppo.py` | **New** |
| 4 | `rsl_rl/rsl_rl/utils/motion_loader.py` | **New** |
| 5 | `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` | **New** |
| 6 | `rsl_rl/rsl_rl/storage/__init__.py` | Edit |
| 7 | `rsl_rl/rsl_rl/modules/__init__.py` | Edit |
| 8 | `rsl_rl/rsl_rl/algorithms/__init__.py` | Edit |
| 9 | `rsl_rl/rsl_rl/utils/__init__.py` | Edit |
| 10 | `rsl_rl/rsl_rl/runners/__init__.py` | Edit |
| 11 | `zouyu/luwu/.../assets/robots/unitree_g1.py` | **New** |
| 12 | `zouyu/luwu/.../tasks/locomotion/amp/__init__.py` | **New** |
| 13 | `zouyu/luwu/.../tasks/locomotion/amp/amp_env_cfg.py` | **New** |
| 14 | `zouyu/luwu/.../tasks/locomotion/amp/mdp/__init__.py` | **New** |
| 15 | `zouyu/luwu/.../tasks/locomotion/amp/mdp/observations.py` | **New** |
| 16 | `zouyu/luwu/.../tasks/locomotion/amp/mdp/rewards.py` | **New** |
| 17 | `zouyu/luwu/.../tasks/locomotion/amp/mdp/terminations.py` | **New** |
| 18 | `zouyu/luwu/.../tasks/locomotion/amp/mdp/events.py` | **New** |
| 19 | `zouyu/luwu/.../tasks/locomotion/amp/config/g1/__init__.py` | **New** |
| 20 | `zouyu/luwu/.../tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py` | **New** |
| 21 | `zouyu/luwu/.../tasks/locomotion/amp/config/g1/agents/__init__.py` | **New** |
| 22 | `zouyu/luwu/.../tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py` | **New** |
| 23 | `zouyu/luwu/scripts/rsl_rl/train_amp.py` | **New** |
| 24 | `zouyu/luwu/.../assets/robots/__init__.py` | Edit |
| 25 | `zouyu/luwu/.../tasks/locomotion/__init__.py` | Edit |

---

## Design Decisions

1. **LSGAN loss** — Following the paper (Section 5.2), MSE loss with +1/-1 targets for stability over sigmoid cross-entropy.
2. **State-only discriminator** — `D(s_t, s_{t+1})` not `D(s_t, a_t)` (Section 5.1), since reference motions don't provide actions.
3. **Gradient penalty on real data only (R1)** — Section 5.4, λ = 10, applied to discriminator observation features.
4. **No backward compatibility** — Clean v5.3 implementation using TensorDict, separate actor/critic models, new RolloutStorage API.
5. **Replay buffer** — Stores policy transitions to prevent discriminator overfitting (Section 6.3).
6. **Reference state initialization** — Episodes start from random reference motion states to improve exploration and imitation quality.
7. **Combined reward** — Linear interpolation between task and style rewards as described in the paper (Equation 4).
8. **No legged_lab dependency** — Task configs directly use IsaacLab base classes (`ManagerBasedRLEnvCfg`). Motion data and animation are handled as extra config attributes on the env cfg.
9. **Full-path imports** — All internal imports use the fully-qualified module path (e.g., `from rsl_rl.storage.replay_buffer import ReplayBuffer`).
10. **Snake-case naming** — Python variables and file names use `snake_case`. Class names use `PascalCase` without ALL_CAPS abbreviations (e.g., `AmpPPO`, `AmpLoader`).
