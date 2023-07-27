import h5py
import argparse
import numpy as np
from tqdm import tqdm

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from waypoint_extraction.extract_waypoints import (
    greedy_waypoint_selection,
    dp_waypoint_selection,
    backtrack_waypoint_selection,
)

num_waypoints = []
num_frames = []


def main(args):
    # create two environments for delta and absolute control, respectively
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    # add linear interpolators for pos and ori
    env_meta["env_kwargs"]["controller_configs"]["interpolation"] = "linear"
    # absolute control
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    env_meta["env_kwargs"]["controller_configs"]["multiplier"] = args.multiplier

    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=True)

    # load the dataset
    f = h5py.File(args.dataset, "r+")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    assert args.start_idx >= 0 and args.end_idx < len(demos)
    for idx in tqdm(range(args.start_idx, args.end_idx + 1), desc="Saving waypoints"):
        ep = demos[idx]

        # prepare initial states to reload from
        states = f[f"data/{ep}/states"][()]
        initial_states = []
        for i in range(len(states)):
            initial_states.append(dict(states=states[i]))
            initial_states[i]["model"] = f[f"data/{ep}"].attrs["model_file"]
        traj_len = states.shape[0]

        # load the ground truth eef pos and rot, joint pos, and gripper qpos
        eef_pos = f[f"data/{ep}/obs/robot0_eef_pos"][()]
        eef_quat = f[f"data/{ep}/obs/robot0_eef_quat"][()]
        joint_pos = f[f"data/{ep}/obs/robot0_joint_pos"][()]
        gt_states = []
        for i in range(traj_len):
            gt_states.append(
                dict(
                    robot0_eef_pos=eef_pos[i],
                    robot0_eef_quat=eef_quat[i],
                    robot0_joint_pos=joint_pos[i],
                )
            )

        # load absolute actions
        try:
            actions = f[f"data/{ep}/abs_actions"][()]
        except:
            print("No absolute actions found, need to convert first.")
            raise NotImplementedError

        # select waypoints automatically
        if args.method == "greedy":
            waypoint_selection = greedy_waypoint_selection
        elif args.method == "dp":
            waypoint_selection = dp_waypoint_selection
        elif args.method == "backtrack":
            waypoint_selection = backtrack_waypoint_selection

        waypoints = waypoint_selection(
            env=env,
            actions=actions,
            gt_states=gt_states,
            err_threshold=args.err_threshold,
            initial_states=initial_states,
            remove_obj=True,
        )

        num_waypoints.append(len(waypoints))
        num_frames.append(traj_len)

        # save waypoints
        try:
            f[f"data/{ep}/waypoints_{args.method}"] = waypoints
        except:
            # if the waypoints dataset already exists, ask the user if they want to overwrite
            print("waypoints dataset already exists. Overwrite? (y/n)")
            ans = input()
            if ans == "y":
                del f[f"data/{ep}/waypoints_{args.method}"]
                f[f"data/{ep}/waypoints_{args.method}"] = waypoints

    f.close()
    print(
        f"Average number of waypoints: {np.mean(num_waypoints)}, average number of frames: {np.mean(num_frames)}, average waypoint ratio: {np.mean(num_frames) / np.mean(num_waypoints)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="robomimic/datasets/lift/ph/low_dim.hdf5",
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

    # method (possible values: greedy, dp, backtrack)
    parser.add_argument(
        "--method",
        type=str,
        default="dp",
        help="(optional) method for waypoint selection",
    )

    # error threshold for reconstructing the trajectory
    parser.add_argument(
        "--err_threshold",
        type=float,
        default=0.01,
        help="(optional) error threshold for reconstructing the trajectory",
    )

    # multiplier for the simulation steps (may need more steps to ensure the robot reaches the goal pose)
    parser.add_argument(
        "--multiplier",
        type=int,
        default=10,
        help="(optional) multiplier for the simulation steps",
    )

    args = parser.parse_args()
    main(args)
