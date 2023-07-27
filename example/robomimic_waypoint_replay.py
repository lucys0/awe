import h5py
import argparse
import imageio
import numpy as np
import time
import wandb
import matplotlib.pyplot as plt

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from utils.utils import plot_3d_trajectory
from waypoint_extraction.traj_reconstruction import reconstruct_waypoint_trajectory
from waypoint_extraction.extract_waypoints import (
    greedy_waypoint_selection,
    dp_waypoint_selection,
    backtrack_waypoint_selection,
    heuristic_waypoint_selection,
)


def main(args):
    # set up wandb
    if args.wandb:
        if args.wandb_run_name is None:
            run_name = args.video_path.split("/")[-1].split(".")[0]
            run_name += f"-{args.task}-idx_{args.start_idx}_{args.end_idx}"
            if args.auto_waypoint:
                run_name += f"-auto_threshold_{args.err_threshold}"
            elif args.constant_waypoint is not None:
                run_name += f"-constant_waypoint_{args.constant_waypoint}"
        else:
            run_name = args.wandb_run_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=args,
        )

    # create two environments for delta and absolute control, respectively
    # this is useful for converting the default delta actions to absolute actions
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
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    assert args.start_idx >= 0 and args.end_idx < len(demos)
    success = []
    num_waypoints = []
    num_frames = []
    traj_err_list = []
    for idx in range(args.start_idx, args.end_idx + 1):
        ep = demos[idx]
        print(f"Playing back episode: {ep}")

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
        vel_ang = f[f"data/{ep}/obs/robot0_eef_vel_ang"][()]
        vel_lin = f[f"data/{ep}/obs/robot0_eef_vel_lin"][()]
        gt_states = []
        for i in range(traj_len):
            gt_states.append(
                dict(
                    robot0_eef_pos=eef_pos[i],
                    robot0_eef_quat=eef_quat[i],
                    robot0_joint_pos=joint_pos[i],
                    robot0_vel_ang=vel_ang[i],
                    robot0_vel_lin=vel_lin[i],
                )
            )

        # load absolute actions
        try:
            if args.diffusion:
                actions = f[f"data/{ep}/actions"][()]
            else:
                actions = f[f"data/{ep}/abs_actions"][()]
        except:
            print("No absolute actions found, need to convert first.")
            raise NotImplementedError

        # add video postfix (task, idx, constant_waypoint / waypoints)
        video_postfix = f"{args.task}-{idx}-"
        assert args.task in args.dataset and args.task in args.video_path

        # set up the waypoint indices
        if args.auto_waypoint:
            if args.preload_auto_waypoint:
                if args.diffusion:
                    waypoint_file = h5py.File(
                        f"robomimic/datasets/{args.task}/ph/low_dim.hdf5", "r"
                    )
                    waypoints = waypoint_file[f"data/{ep}/waypoints_dp"][()]
                    # increase waypoints by 1 except the last one
                    waypoints = np.concatenate(
                        [waypoints[:-1] + 1, waypoints[-1:]], axis=0
                    )
                else:
                    waypoints = f[f"data/{ep}/waypoints_dp"][()]
                print(f"Preloaded waypoints: {waypoints}")
            else:
                # select waypoints automatically
                start_time = time.time()
                waypoints = dp_waypoint_selection(
                    env=env,
                    actions=actions,
                    gt_states=gt_states,
                    err_threshold=args.err_threshold,
                    initial_states=initial_states,
                    remove_obj=True,
                )
                total_time = time.time() - start_time
                print(f"Automatic waypoint selection took {total_time:.2f} seconds")
                if args.wandb:
                    wandb.log({"time/auto_waypoint_selection": total_time}, step=idx)

            video_postfix += (
                "auto_waypoint"
                + f"_err_{args.err_threshold}_"
                + "_".join([str(k) for k in waypoints])
            )
        elif args.waypoints is None:
            constant_waypoint = (
                args.constant_waypoint if args.constant_waypoint is not None else 1
            )
            waypoints = np.arange(1, traj_len, constant_waypoint)
            # add the last step if not already present
            if waypoints[-1] != traj_len - 1:
                waypoints = np.append(waypoints, traj_len - 1)
            if constant_waypoint != 1:
                video_postfix += f"constant_waypoint_{constant_waypoint}"
        else:
            # parse as a list of integers
            waypoints = [int(k) for k in args.waypoints.split(",")]
            # convert waypoints list to string for video filename
            video_postfix += "waypoints_" + "_".join([str(k) for k in waypoints])

        # create a video writer
        video_path = args.video_path.replace(".mp4", f"-{video_postfix}.mp4")
        if args.record_video:
            print(f"Generating video for task {args.task} on data idx {idx}")
            video_writer = imageio.get_writer(video_path, fps=20)
        else:
            video_writer = None

        # recreate the trajectory by following waypoints
        start_time = time.time()
        pred_states_list, _, traj_err = reconstruct_waypoint_trajectory(
            env=env,
            actions=actions,
            gt_states=gt_states,
            waypoints=waypoints,
            video_writer=video_writer,
            verbose=True,
            initial_state=initial_states[0],
            remove_obj=args.remove_object,
        )
        total_time = time.time() - start_time
        print(f"Simulation took {total_time:.2f} seconds")

        num_waypoint = len(waypoints)
        print(f"Number of waypoints: {num_waypoint}")
        num_waypoints.append(num_waypoint)
        traj_err_list.append(traj_err)
        num_frames.append(len(actions))

        if args.wandb:
            wandb.log({"time/simulation": total_time}, step=idx)
            wandb.log({"num_waypoints": num_waypoint}, step=idx)
            wandb.log({"traj_err": traj_err}, step=idx)

        # check if successful
        if not args.remove_object:
            is_success = env.is_success()["task"]
            print(f"Success at ep {idx}: {is_success}")
            success.append(is_success)
            if args.wandb:
                wandb.log({"success": int(is_success)}, step=idx)
                wandb.log({"avg_success_rate_sofar": np.mean(success)})

        # record a 3D visualization
        if args.plot_3d:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"Task: {args.task}, Data idx: {idx}")

            plot_3d_trajectory(
                ax,
                [s["robot0_eef_pos"] for s in gt_states],
                label="gt",
                gripper=actions[:, -1],
            )
            plot_3d_trajectory(
                ax, pred_states_list, label="pred", gripper=actions[:, -1]
            )
            # plot_3d_trajectory(ax, actions[1:, :3], label="action", gripper=actions[:, -1])

            fig.savefig(video_path.replace(".mp4", ".png"))
            if args.wandb:
                wandb.log({"3d_traj": wandb.Image(fig)}, step=idx)
            plt.close(fig)

        if args.record_video:
            video_writer.close()

    if args.wandb:
        wandb.log({"avg_num_waypoints": np.mean(num_waypoints)})
        wandb.log({"avg_traj_err": np.mean(traj_err_list)})
        wandb.log({"avg_num_frames": np.mean(num_frames)})

    # compute the success rate
    if not args.remove_object:
        avg_success_rate = np.mean(success)
        print(f"Success rate: {avg_success_rate}")
        if args.wandb:
            wandb.log({"avg_success_rate": avg_success_rate})

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    # task name
    parser.add_argument(
        "--task",
        type=str,
        default="lift",
        help="task name",
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
        default=2,
        help="(optional) end index of the trajectory to playback",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default="agentview",
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # list of waypoints, default to None
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        help="(optional) list of waypoints to recreate the trajectory",
    )

    # constant interval between waypoints
    parser.add_argument(
        "--constant_waypoint",
        type=int,
        default=None,
        help="(optional) constant interval between waypoints",
    )

    # whether to record the video
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="(optional) whether to record the video",
    )

    # whether to remove object
    parser.add_argument(
        "--remove_object",
        action="store_true",
        help="(optional) whether to remove objects",
    )

    # whether to plot the 3D trajectory
    parser.add_argument(
        "--plot_3d",
        action="store_true",
        help="(optional) whether to plot the 3D trajectory",
    )

    # whether to select waypoints automatically
    parser.add_argument(
        "--auto_waypoint",
        action="store_true",
        help="(optional) whether to select waypoints automatically",
    )

    # whether to preload auto waypoints
    parser.add_argument(
        "--preload_auto_waypoint",
        action="store_true",
        help="(optional) whether to preload auto waypoints",
    )

    # error threshold for reconstructing the trajectory
    parser.add_argument(
        "--err_threshold",
        type=float,
        default=0.03,
        help="(optional) error threshold for reconstructing the trajectory",
    )

    # multiplier for the simulation steps (may need more steps to ensure the robot reaches the goal pose)
    parser.add_argument(
        "--multiplier",
        type=int,
        default=10,
        help="(optional) multiplier for the simulation steps",
    )

    # wandb
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="(optional) whether to use wandb",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="(optional) wandb entity",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="awe",
        help="(optional) wandb project",
    )

    # wandb run name
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="(optional) wandb run name",
    )

    # diffusion
    parser.add_argument(
        "--diffusion",
        action="store_true",
        help="(optional) whether to use the diffusion dataset",
    )

    args = parser.parse_args()
    main(args)
