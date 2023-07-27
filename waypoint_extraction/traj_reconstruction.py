import numpy as np
import wandb
from scipy.spatial.transform import Rotation

from utils import put_text, remove_object

import robosuite.utils.transform_utils as T


def linear_interpolation(p1, p2, t):
    """Compute the linear interpolation between two 3D points"""
    return p1 + t * (p2 - p1)


def point_line_distance(point, line_start, line_end):
    """Compute the shortest distance between a 3D point and a line segment defined by two 3D points"""
    line_vector = line_end - line_start
    point_vector = point - line_start
    # t represents the position of the orthogonal projection of the given point onto the infinite line defined by the segment
    t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    t = np.clip(t, 0, 1)
    projection = linear_interpolation(line_start, line_end, t)
    return np.linalg.norm(point - projection)


def point_quat_distance(point, quat_start, quat_end, t, total):
    pred_point = T.quat_slerp(quat_start, quat_end, fraction=t / total)
    err_quat = (
        Rotation.from_quat(pred_point) * Rotation.from_quat(point).inv()
    ).magnitude()
    return err_quat


def geometric_waypoint_trajectory(actions, gt_states, waypoints, return_list=False):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    gt_pos = [p["robot0_eef_pos"] for p in gt_states]
    gt_quat = [p["robot0_eef_quat"] for p in gt_states]

    keypoints_pos = [actions[k, :3] for k in waypoints]
    keypoints_quat = [gt_quat[k] for k in waypoints]

    state_err = []

    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]
        start_keypoint_quat = keypoints_quat[i]
        end_keypoint_quat = keypoints_quat[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_pos[start_idx:end_idx]
        segment_points_quat = gt_quat[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            rot_err = point_quat_distance(
                segment_points_quat[i],
                start_keypoint_quat,
                end_keypoint_quat,
                i,
                len(segment_points_quat),
            )
            state_err.append(pos_err + rot_err)

    # print the average and max error for pos and rot
    # print(f"Average pos error: {np.mean(pos_err_list):.6f} \t Average rot error: {np.mean(rot_err_list):.6f}")
    # print(f"Max pos error: {np.max(pos_err_list):.6f} \t Max rot error: {np.max(rot_err_list):.6f}")

    if return_list:
        return total_traj_err(state_err), state_err
    return total_traj_err(state_err)


def pos_only_geometric_waypoint_trajectory(
    actions, gt_states, waypoints, return_list=False
):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints

    keypoints_pos = [actions[k] for k in waypoints]
    state_err = []
    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_states[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            state_err.append(pos_err)

    # print the average and max error
    print(
        f"Average pos error: {np.mean(state_err):.6f} \t Max pos error: {np.max(state_err):.6f}"
    )

    if return_list:
        return total_traj_err(state_err), state_err
    else:
        return total_traj_err(state_err)


def reconstruct_waypoint_trajectory(
    env,
    actions,
    gt_states,
    waypoints,
    initial_state=None,
    video_writer=None,
    verbose=True,
    remove_obj=False,
):
    """Reconstruct the trajectory from a set of waypoints"""
    curr_frame = 0
    pred_states_list = []
    traj_err_list = []
    frames = []
    total_DTW_distance = 0

    if len(gt_states) <= 1:
        return pred_states_list, traj_err_list, 0

    # reset the environment
    reset_env(env=env, initial_state=initial_state, remove_obj=remove_obj)
    prev_k = 0

    for k in waypoints:
        # skip to the next keypoint
        if verbose:
            print(f"Next keypoint: {k}")
        _, _, _, info = env.step(actions[k], skip=k - prev_k, record=True)
        prev_k = k

        # select a subset of states that are the closest to the ground truth
        DTW_distance, pred_states = dynamic_time_warping(
            gt_states[curr_frame + 1 : k + 1], info["intermediate_states"]
        )
        if verbose:
            print(
                f"selected subsequence: {pred_states} with DTW distance: {DTW_distance}"
            )
        total_DTW_distance += DTW_distance
        for i in range(len(pred_states)):
            # measure the error between the recorded and actual states
            curr_frame += 1
            pred_state_idx = pred_states[i]
            err_dict = compute_state_error(
                gt_states[curr_frame], info["intermediate_states"][pred_state_idx]
            )
            traj_err_list.append(total_state_err(err_dict))
            pred_states_list.append(
                info["intermediate_states"][pred_state_idx]["robot0_eef_pos"]
            )
            # save the frames (agentview)
            if video_writer is not None:
                frame = put_text(
                    info["intermediate_obs"][pred_state_idx],
                    curr_frame,
                    is_waypoint=(k == curr_frame),
                )
                video_writer.append_data(frame)
                if wandb.run is not None:
                    frames.append(frame)

    # wandb log the video if initialized
    if video_writer is not None and wandb.run is not None:
        video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
        wandb.log({"video": wandb.Video(video, fps=20, format="mp4")})

    # compute the total trajectory error
    traj_err = total_traj_err(traj_err_list)
    if verbose:
        print(
            f"Total trajectory error: {traj_err:.6f} \t Total DTW distance: {total_DTW_distance:.6f}"
        )

    return pred_states_list, traj_err_list, traj_err


def total_state_err(err_dict):
    return err_dict["err_pos"] + err_dict["err_quat"]


def total_traj_err(err_list):
    # return np.mean(err_list)
    return np.max(err_list)


def compute_state_error(gt_state, pred_state):
    """Compute the state error between the ground truth and predicted states."""
    err_pos = np.linalg.norm(gt_state["robot0_eef_pos"] - pred_state["robot0_eef_pos"])
    err_quat = (
        Rotation.from_quat(gt_state["robot0_eef_quat"])
        * Rotation.from_quat(pred_state["robot0_eef_quat"]).inv()
    ).magnitude()
    err_joint_pos = np.linalg.norm(
        gt_state["robot0_joint_pos"] - pred_state["robot0_joint_pos"]
    )
    state_err = dict(err_pos=err_pos, err_quat=err_quat, err_joint_pos=err_joint_pos)
    return state_err


def dynamic_time_warping(seq1, seq2, idx1=0, idx2=0, memo=None):
    if memo is None:
        memo = {}

    if idx1 == len(seq1):
        return 0, []

    if idx2 == len(seq2):
        return float("inf"), []

    if (idx1, idx2) in memo:
        return memo[(idx1, idx2)]

    distance_with_current = total_state_err(compute_state_error(seq1[idx1], seq2[idx2]))
    error_with_current, subseq_with_current = dynamic_time_warping(
        seq1, seq2, idx1 + 1, idx2 + 1, memo
    )
    error_with_current += distance_with_current

    error_without_current, subseq_without_current = dynamic_time_warping(
        seq1, seq2, idx1, idx2 + 1, memo
    )

    if error_with_current < error_without_current:
        memo[(idx1, idx2)] = error_with_current, [idx2] + subseq_with_current
    else:
        memo[(idx1, idx2)] = error_without_current, subseq_without_current

    return memo[(idx1, idx2)]


def reset_env(env, initial_state=None, remove_obj=False):
    # load the initial state
    if initial_state is not None:
        env.reset_to(initial_state)

    # remove the object from the scene
    if remove_obj:
        remove_object(env)
