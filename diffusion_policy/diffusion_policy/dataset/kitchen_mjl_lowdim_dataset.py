from typing import Dict
import torch
import numpy as np
import copy
import pathlib
import h5py
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env.kitchen.kitchen_util import parse_mjl_logs


def relabel_waypoints(arr, waypoint_indices):
    start_idx = 0
    for key_idx in waypoint_indices:
        # Replace the items between the start index and the key index with the key item
        arr[start_idx:key_idx] = arr[key_idx]
        start_idx = key_idx
    return arr


class KitchenMjlLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=True,
        robot_noise_ratio=0.0,
        seed=42,
        val_ratio=0.0,
        use_waypoint=False,
        constant_waypoint=None,
    ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed=seed)

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()

        if use_waypoint:
            waypoint_path = data_directory.parent / "waypoints.hdf5"
            waypoint_file = h5py.File(waypoint_path, "r")

        for i, mjl_path in enumerate(tqdm(list(data_directory.glob("*/*.mjl")))):
            try:
                data = parse_mjl_logs(
                    str(mjl_path.absolute()), skipamount=40
                )  # XXX: skipamount?
                qpos = data["qpos"].astype(np.float32)
                obs = np.concatenate(
                    [
                        qpos[:, :9],
                        qpos[:, -21:],
                        np.zeros((len(qpos), 30), dtype=np.float32),
                    ],
                    axis=-1,
                )
                if robot_noise_ratio > 0:
                    # add observation noise to match real robot
                    noise = (
                        robot_noise_ratio
                        * robot_pos_noise_amp
                        * rng.uniform(low=-1.0, high=1.0, size=(obs.shape[0], 30))
                    )
                    obs[:, :30] += noise
                action = data["ctrl"].astype(np.float32)
                if use_waypoint:
                    if constant_waypoint is not None:
                        traj_len = action.shape[0]
                        waypoints = np.arange(1, traj_len, constant_waypoint)
                        if (
                            waypoints[-1] != traj_len - 1
                        ):  # add the last step if not already present
                            waypoints = np.append(waypoints, traj_len - 1)
                    else:
                        waypoints = waypoint_file[f"greedy/{i}"][()]
                        if i == 0:
                            print(
                                f"Episode {i}: {len(waypoints)} waypoints, {waypoints}"
                            )

                    action = relabel_waypoints(action, waypoints)
                episode = {"obs": obs, "action": action}
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "obs": self.replay_buffer["obs"],
            "action": self.replay_buffer["action"],
        }
        if "range_eps" not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs["range_eps"] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
