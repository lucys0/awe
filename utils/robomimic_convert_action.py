import h5py
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


def main(args):
    # create environment (delta control)
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=True)

    # load the dataset
    f = h5py.File(args.dataset, "r+")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    assert args.start_idx >= 0 and args.end_idx < len(demos)

    for idx in tqdm(range(args.start_idx, args.end_idx + 1), desc="Converting actions"):
        ep = demos[idx]
        states = f[f"data/{ep}/states"][()]
        traj_len = states.shape[0]

        # generate abs actions
        delta_actions = f[f"data/{ep}/actions"][()]
        action_pos = np.zeros((traj_len, 3), dtype=delta_actions.dtype)
        action_ori = np.zeros((traj_len, 3), dtype=delta_actions.dtype)
        action_gripper = delta_actions[:, -1:]

        # record low dim states
        obs = env.reset_to({"states": states[0]})
        # convert to list
        for k in obs.keys():
            obs[k] = [obs[k]]

        robot = env.env.robots[0]
        controller = robot.controller

        first = True

        for i in range(len(states)):
            env.reset_to({"states": states[i]})
            # run controller
            robot.control(delta_actions[i], policy_step=True)

            if first:
                initial_state = env.get_state()["states"]
                first = False

            # read pos and ori from robots
            action_pos[i] = controller.ee_pos
            action_ori[i] = Rotation.from_matrix(controller.ee_ori_mat).as_rotvec()

            # record low dim states
            new_obs = env.get_observation()
            for k in obs.keys():
                obs[k].append(new_obs[k])
        actions = np.concatenate([action_pos, action_ori, action_gripper], axis=-1)
        # stack obs
        for k in obs.keys():
            obs[k] = np.stack(obs[k], axis=0)

        # dump into a file of abs_actions in the original dataset
        f[f"data/{ep}/abs_actions"] = actions

        # dump into a file of abs_obs in the original dataset
        for k in obs.keys():
            f[f"data/{ep}/abs_obs/{k}"] = obs[k]
        f[f"data/{ep}/initial_state"] = initial_state

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    # index of the trajectory to playback. If omitted, playback trajectory 0.
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="(optional) start index of the trajectory to playback",
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=199,
        help="(optional) end index of the trajectory to playback",
    )

    args = parser.parse_args()
    main(args)
