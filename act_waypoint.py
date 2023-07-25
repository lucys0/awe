import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.utils import plot_3d_trajectory
from waypoint_selection import dp_waypoint_selection, greedy_waypoint_selection


def main(args):
    num_waypoints = []
    num_frames = []

    # load data
    for i in tqdm(range(args.start_idx, args.end_idx + 1)):
        dataset_path = os.path.join(args.dataset, f"episode_{i}.hdf5")
        with h5py.File(dataset_path, "r+") as root:
            qpos = root["/observations/qpos"][()]

            if args.use_ee:
                qpos = np.array(qpos)  # ts, dim

                # calculate EE pose
                from act.convert_ee import get_ee

                left_arm_ee = get_ee(qpos[:, :6], qpos[:, 6:7])
                right_arm_ee = get_ee(qpos[:, 7:13], qpos[:, 13:14])
                qpos = np.concatenate([left_arm_ee, right_arm_ee], axis=1)

            # select waypoints
            waypoints = dp_waypoint_selection( # if it's too slow, use greedy_waypoint_selection
                env=None,
                actions=qpos,
                gt_states=qpos,
                err_threshold=args.err_threshold,
                pos_only=True,
            )
            print(
                f"Episode {i}: {len(qpos)} frames -> {len(waypoints)} waypoints (ratio: {len(qpos)/len(waypoints):.2f})"
            )
            num_waypoints.append(len(waypoints))
            num_frames.append(len(qpos))

            # save waypoints
            if args.save_waypoints:
                name = f"/waypoints"
                try:
                    root[name] = waypoints
                except:
                    # if the waypoints dataset already exists, ask the user if they want to overwrite
                    print("waypoints dataset already exists. Overwrite? (y/n)")
                    ans = input()
                    if ans == "y":
                        del root[name]
                        root[name] = waypoints

            # visualize ground truth qpos and waypoints
            if args.plot_3d:
                if not args.use_ee:
                    qpos = np.array(qpos)  # ts, dim
                    from act.convert_ee import get_xyz

                    left_arm_xyz = get_xyz(qpos[:, :6])
                    right_arm_xyz = get_xyz(qpos[:, 7:13])
                else:
                    left_arm_xyz = left_arm_ee[:, :3]
                    right_arm_xyz = right_arm_ee[:, :3]

                # Find global min and max for each axis
                all_data = np.concatenate([left_arm_xyz, right_arm_xyz], axis=0)
                min_x, min_y, min_z = np.min(all_data, axis=0)
                max_x, max_y, max_z = np.max(all_data, axis=0)

                fig = plt.figure(figsize=(20, 10))
                ax1 = fig.add_subplot(121, projection="3d") 
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.set_zlabel("z")
                ax1.set_title("Left", fontsize=20)
                ax1.set_xlim([min_x, max_x])
                ax1.set_ylim([min_y, max_y])
                ax1.set_zlim([min_z, max_z])

                plot_3d_trajectory(ax1, left_arm_xyz, label="ground truth", legend=False)

                ax2 = fig.add_subplot(122, projection="3d")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                ax2.set_zlabel("z")
                ax2.set_title("Right", fontsize=20)
                ax2.set_xlim([min_x, max_x])
                ax2.set_ylim([min_y, max_y])
                ax2.set_zlim([min_z, max_z])

                plot_3d_trajectory(ax2, right_arm_xyz, label="ground truth", legend=False)

                # prepend 0 to waypoints to include the initial state
                waypoints = [0] + waypoints

                plot_3d_trajectory(
                    ax1,
                    [left_arm_xyz[i] for i in waypoints],
                    label="waypoints",
                    legend=False,
                )  # Plot waypoints for left_arm_xyz
                plot_3d_trajectory(
                    ax2,
                    [right_arm_xyz[i] for i in waypoints],
                    label="waypoints",
                    legend=False,
                )  # Plot waypoints for right_arm_xyz

                fig.suptitle(f"Task: {args.dataset.split('/')[-1]}", fontsize=30) 

                handles, labels = ax1.get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)

                fig.savefig(
                    f"plot/act/{args.dataset.split('/')[-1]}_{i}_t_{args.err_threshold}_waypoints.png"
                )
                plt.close(fig)

            root.close()

    print(
        f"Average number of waypoints: {np.mean(num_waypoints)} \tAverage number of frames: {np.mean(num_frames)} \tratio: {np.mean(num_frames)/np.mean(num_waypoints)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/act/sim_transfer_cube_scripted",
        # default="data/act/sim_insertion_scripted",
        # default="data/act/sim_transfer_cube_human",
        # default="data/act/sim_insertion_human",
        # default="data/act/aloha_screw_driver",
        # default="data/act/aloha_coffee",
        # default="data/act/aloha_towel",
        # default="data/act/aloha_coffee_new",
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
        default=49,
        help="(optional) end index of the trajectory to playback",
    )

    # error threshold for reconstructing the trajectory
    parser.add_argument(
        "--err_threshold",
        type=float,
        default=0.05,
        help="(optional) error threshold for reconstructing the trajectory",
    )

    # whether to save waypoints
    parser.add_argument(
        "--save_waypoints",
        action="store_true",
        help="(optional) whether to save waypoints",
    )

    # whether to use the ee space for waypoint selection
    parser.add_argument(
        "--use_ee",
        action="store_true",
        help="(optional) whether to use the ee space for waypoint selection",
    )

    # whether to plot 3d
    parser.add_argument(
        "--plot_3d",
        action="store_true",
        help="(optional) whether to plot 3d",
    )

    args = parser.parse_args()
    main(args)
