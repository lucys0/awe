import h5py
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from waypoint_extraction.extract_waypoints import (
    greedy_waypoint_selection,
    dp_waypoint_selection,
    backtrack_waypoint_selection,
)
from utils import plot_3d_trajectory

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

    # the third term in args.dataset is the task name
    task = args.dataset.split("/")[2]

    assert args.start_idx >= 0 and args.end_idx < len(demos)
    for idx in tqdm(
        range(args.start_idx, args.end_idx + 1), desc="Visualizing waypoints"
    ):
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

        fig = plt.figure(
            figsize=(10 * len(args.err_threshold), 10)
        )  # adjusted size based on the number of thresholds

        for i, err_thresh in enumerate(args.err_threshold):
            ax = fig.add_subplot(1, len(args.err_threshold), i + 1, projection="3d")

            waypoints = waypoint_selection(
                env=env,
                actions=actions,
                gt_states=gt_states,
                err_threshold=err_thresh,
                initial_states=initial_states,
                remove_obj=True,
            )

            num_waypoints.append(len(waypoints))
            num_frames.append(traj_len)

            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            # remove the ticks
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])
            # remove the tick labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            ax.set_title(f"Error budget = {err_thresh}", fontsize=26)

            gt_pos = [s["robot0_eef_pos"] for s in gt_states]
            plot_3d_trajectory(ax, gt_pos, label="ground truth", legend=False)

            # waypoint_states is the slice of gt_pos that corresponds to the waypoints
            # prepend 0 to waypoints to include the initial state
            waypoints = [0] + waypoints
            waypoint_states = [gt_pos[i] for i in waypoints]

            plot_3d_trajectory(ax, waypoint_states, label="waypoints", legend=False)

        fig.suptitle(
            f"Task: {task}", fontsize=30
        )  # set a common title for all subplots

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=2, fontsize=26
        )  # larger font size for legend, adjusted position
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust the subplot margins
        # fig.tight_layout(rect=[0, 0.03, 1, 0.8])  # adjust the subplot margins
        # fig.subplots_adjust(bottom=0.2, top=0.8)
        # reduce the margin between the subplots
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        # reduce left and right margins
        fig.subplots_adjust(left=0.05, right=0.95)

        fig.savefig(f"plot/epsilon/waypoint_{task}.png")
        plt.close(fig)

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="robomimic/datasets/can/ph/low_dim.hdf5",
        help="path to hdf5 dataset",
    )

    # index of the trajectory to playback. If omitted, playback trajectory 0.
    parser.add_argument(
        "--start_idx",
        type=int,
        default=1,
        help="(optional) start index of the trajectory to playback",
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=1,
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
        nargs="+",  # updated to accept list of floats
        default=[0.01, 0.005],  # default is list of thresholds
        help="(optional) error thresholds for reconstructing the trajectory",
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
