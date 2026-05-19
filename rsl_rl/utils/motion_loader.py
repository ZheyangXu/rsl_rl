# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class MotionData:
    """Processed motion data for a single motion clip.

    Stores joint positions/velocities, base velocities (world and local frames),
    and base orientation as a quaternion. All quantities are torch tensors on the
    specified device.

    Quaternions are stored in wxyz format (IsaacLab convention), converted
    from the xyzw format (SciPy default) during loading.
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor]
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        """Convert numpy arrays and SciPy rotations to torch tensors."""
        def _to_tensor(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = _to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = _to_tensor(self.joint_velocities)
        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = _to_tensor(self.base_lin_velocities_mixed)
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = _to_tensor(self.base_ang_velocities_mixed)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = _to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = _to_tensor(self.base_ang_velocities_local)
        if isinstance(self.base_quat, Rotation):
            quat_xyzw = self.base_quat.as_quat()
            self.base_quat = _to_tensor(quat_xyzw[:, [3, 0, 1, 2]])

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the concatenated AMP observation for given indices.

        Order: joint_positions, joint_velocities, base_lin_velocities_local,
               base_ang_velocities_local.

        Args:
            indices: Sample indices to retrieve.

        Returns:
            Concatenated observation tensor (len(indices), total_dim).
        """
        return torch.cat(
            (
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_local[indices],
                self.base_ang_velocities_local[indices],
            ),
            dim=1,
        )

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return the full state for environment reset.

        Args:
            indices: Sample indices to retrieve.

        Returns:
            Tuple of (quat, joint_positions, joint_velocities,
                      base_lin_velocities, base_ang_velocities).
        """
        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )


class AmpLoader:
    """Loader for motion capture datasets in AMP-compatible format.

    Handles loading, joint reordering, temporal resampling to match the
    simulator timestep, velocity computation, coordinate frame conversion,
    and precomputed sampling buffers for efficient training.

    Dataset format (per file, stored as .npy or .pkl):
        {
            "joints_list": List[str],
            "joint_positions": List[np.ndarray],
            "root_position": List[np.ndarray],
            "root_quaternion": List[np.ndarray] (xyzw, SciPy convention),
            "fps": float,
        }

    Args:
        device: Target torch device.
        dataset_path_root: Directory containing the motion data files.
        datasets: Mapping from dataset name (without extension) to
            sampling weight (float).
        simulation_dt: Timestep used by the simulator (sim.dt * decimation).
        slow_down_factor: Factor to slow down the original motion data.
        expected_joint_names: Override for joint ordering. If None, the union
            of all joint names across datasets is used.
    """

    def __init__(
        self,
        device: str | torch.device,
        dataset_path_root: str | Path,
        datasets: Dict[str, float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
    ) -> None:
        self.device = device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        dataset_names = list(datasets.keys())
        dataset_weights_list = list(datasets.values())

        # Build union of joint names across all datasets if not provided.
        # For .pkl format (no joints_list), expected_joint_names must be
        # supplied externally.
        if expected_joint_names is None:
            joint_union: List[str] = []
            seen = set()
            found_joints = False
            for name in dataset_names:
                data_path = _resolve_path(dataset_path_root, name)
                info = _load_data_dict(data_path)
                if "joints_list" in info:
                    found_joints = True
                    for j in info["joints_list"]:
                        if j not in seen:
                            seen.add(j)
                            joint_union.append(j)
            if found_joints:
                expected_joint_names = joint_union

        # Load and process each dataset
        self.motion_data: List[MotionData] = []
        for dataset_name in dataset_names:
            dataset_path = _resolve_path(dataset_path_root, dataset_name)
            md = self._load_data(
                dataset_path,
                simulation_dt,
                slow_down_factor,
                expected_joint_names,
            )
            self.motion_data.append(md)

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weights_list, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list, reset_states = [], [], []
        for data, w in zip(self.motion_data, self.dataset_weights):
            t_len = len(data)
            idx = torch.arange(t_len, device=self.device)
            obs = data.get_amp_dataset_obs(idx)
            next_idx = torch.clamp(idx + 1, max=t_len - 1)
            next_obs = data.get_amp_dataset_obs(next_idx)

            obs_list.append(obs)
            next_obs_list.append(next_obs)

            quat, jp, jv, blv, bav = data.get_state_for_reset(idx)
            reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        # Per-frame sampling weights: weight_i / length_i
        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yield mini-batches of (state, next_state) pairs from expert data.

        Args:
            num_mini_batch: Number of mini-batches to yield.
            mini_batch_size: Number of samples per mini-batch.

        Yields:
            Tuple of (state_batch, next_state_batch) tensors.
        """
        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """Randomly sample full states for reference state initialization.

        Args:
            number_of_samples: Number of states to sample.

        Returns:
            Tuple of (quat, joint_positions, joint_velocities,
                      base_lin_velocities, base_ang_velocities).
        """
        idx = torch.multinomial(
            self.per_frame_weights, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        joint_dim = self.motion_data[0].joint_positions.shape[1]
        dims = [4, joint_dim, joint_dim, 3, 3]
        return tuple(torch.split(full, dims, dim=1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_rn(
        data: np.ndarray,
        original_keyframes: np.ndarray,
        target_keyframes: np.ndarray,
    ) -> np.ndarray:
        f = interp1d(original_keyframes, data, axis=0)
        return f(target_keyframes)

    @staticmethod
    def _resample_so3(
        raw_quaternions: np.ndarray,
        original_keyframes: np.ndarray,
        target_keyframes: np.ndarray,
    ) -> Rotation:
        tmp = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, tmp)
        return slerp(target_keyframes)

    @staticmethod
    def _compute_ang_vel(
        data: Rotation,
        dt: float,
        local: bool = False,
    ) -> np.ndarray:
        r_prev = data[:-1]
        r_next = data[1:]
        if local:
            rel = r_prev.inv() * r_next
        else:
            rel = r_next * r_prev.inv()
        rotvec = rel.as_rotvec() / dt
        return np.vstack((rotvec, rotvec[-1]))

    @staticmethod
    def _compute_raw_derivative(data: np.ndarray, dt: float) -> np.ndarray:
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    def _load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None],
    ) -> MotionData:
        """Load and process a single motion dataset file.

        Supports two formats:
        - .npy: Contains ``joints_list``, ``joint_positions`` (List[ndarray]),
          ``root_position`` (List[ndarray]), ``root_quaternion`` (List[ndarray]).
        - .pkl: Contains ``dof_pos`` (ndarray T×N), ``root_pos`` (ndarray T×3),
          ``root_rot`` (ndarray T×4, xyzw), with no ``joints_list``.
        """
        data = _load_data_dict(dataset_path)

        if "joints_list" in data:
            return self._load_npy_format(
                data, simulation_dt, slow_down_factor, expected_joint_names
            )
        else:
            return self._load_pkl_format(
                data, simulation_dt, slow_down_factor
            )

    def _load_pkl_format(
        self,
        data: dict,
        simulation_dt: float,
        slow_down_factor: int,
    ) -> MotionData:
        """Load .pkl format: dof_pos (T,N), root_pos (T,3), root_rot (T,4 xyzw)."""
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_rot = np.asarray(data["root_rot"], dtype=np.float32)

        dt = 1.0 / data["fps"] / float(slow_down_factor)
        t_len = dof_pos.shape[0]
        t_orig = np.linspace(0, t_len * dt, t_len)
        t_new_len = int(t_len * dt / simulation_dt)
        if t_new_len < 2:
            t_new_len = 2
        t_new = np.linspace(0, t_len * dt, t_new_len)

        resampled_joint_positions = self._resample_rn(dof_pos, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_rn(root_pos, t_orig, t_new)
        resampled_base_orientations = self._resample_so3(
            root_rot, t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )
        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )

        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(resampled_base_orientations, resampled_base_lin_vel_mixed)
            ]
        )
        resampled_base_ang_vel_local = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=True
        )

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,
            device=self.device,
        )

    def _load_npy_format(
        self,
        data: dict,
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None],
    ) -> MotionData:
        """Load .npy format with joints_list + per-frame joint_positions."""
        dataset_joint_names = data["joints_list"]
        if expected_joint_names is None:
            expected_joint_names = dataset_joint_names

        # Build index map for expected_joint_names
        idx_map: List[Union[int, None]] = []
        for j in expected_joint_names:
            if j in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(j))
            else:
                idx_map.append(None)

        # Reorder and fill joint positions
        jp_list: List[np.ndarray] = []
        for frame in data["joint_positions"]:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)

        dt = 1.0 / data["fps"] / float(slow_down_factor)
        t_len = len(jp_list)
        t_orig = np.linspace(0, t_len * dt, t_len)
        t_new_len = int(t_len * dt / simulation_dt)
        if t_new_len < 2:
            t_new_len = 2
        t_new = np.linspace(0, t_len * dt, t_new_len)

        resampled_joint_positions = self._resample_rn(np.asarray(jp_list), t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_rn(
            np.asarray(data["root_position"]), t_orig, t_new
        )
        resampled_base_orientations = self._resample_so3(
            data["root_quaternion"], t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )
        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )

        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(resampled_base_orientations, resampled_base_lin_vel_mixed)
            ]
        )
        resampled_base_ang_vel_local = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=True
        )

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,
            device=self.device,
        )


def _load_data_dict(path: Path) -> dict:
    """Load a dataset file (.npy or .pkl) and return its dict content."""
    data = np.load(str(path), allow_pickle=True)
    if isinstance(data, dict):
        return data
    if hasattr(data, "item"):
        return data.item()
    # .pkl files with ndarray-only content
    import pickle
    with open(str(path), "rb") as fh:
        return pickle.load(fh)


def _resolve_path(root: Path, name: str) -> Path:
    """Resolve a dataset path, trying .npy first, then .pkl."""
    for ext in (".npy", ".pkl"):
        candidate = root / f"{name}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset not found: {root / name}.npy or .pkl")
