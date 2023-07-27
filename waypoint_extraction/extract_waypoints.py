""" Automatic waypoint selection """
import numpy as np
import copy

from waypoint_extraction.traj_reconstruction import (
    pos_only_geometric_waypoint_trajectory,
    reconstruct_waypoint_trajectory,
    geometric_waypoint_trajectory,
)


""" Iterative waypoint selection """
def greedy_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    geometry=True,
    pos_only=False,
):
    # make the last frame a waypoint
    waypoints = [len(actions) - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(len(actions) - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                waypoints.append(i)
                waypoints.append(i + 1)
        waypoints.sort()

    # reconstruct the trajectory, and record the reconstruction error for each state
    for i in range(len(actions)):
        if pos_only or geometry:
            func = (
                pos_only_geometric_waypoint_trajectory
                if pos_only
                else geometric_waypoint_trajectory
            )
            total_traj_err, reconstruction_error = func(
                actions=actions,
                gt_states=gt_states,
                waypoints=waypoints,
                return_list=True,
            )
        else:
            _, reconstruction_error, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions,
                gt_states=gt_states,
                waypoints=waypoints,
                verbose=False,
                initial_state=initial_states[0],
                remove_obj=remove_obj,
            )
        # break if the reconstruction error is below the threshold
        if total_traj_err < err_threshold:
            break
        # add the frame of the highest reconstruction error as a waypoint, excluding frames that are already waypoints
        max_error_frame = np.argmax(reconstruction_error)
        while max_error_frame in waypoints:
            reconstruction_error[max_error_frame] = 0
            max_error_frame = np.argmax(reconstruction_error)
        waypoints.append(max_error_frame)
        waypoints.sort()

    print("=======================================================================")
    print(
        f"Selected {len(waypoints)} waypoints: {waypoints} \t total trajectory error: {total_traj_err:.6f}"
    )
    return waypoints


def heuristic_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    geometry=True,
    pos_only=False,
):
    # make the last frame a waypoint
    waypoints = [len(actions) - 1]

    # make the frames of gripper open/close waypoints
    for i in range(len(actions) - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            waypoints.append(i)
    waypoints.sort()

    # if 'robot0_vel_ang' or 'robot0_vel_lin' in gt_states is close to 0, make the frame a waypoint
    for i in range(len(gt_states)):
        if (
            np.linalg.norm(gt_states[i]["robot0_vel_ang"]) < err_threshold
            or np.linalg.norm(gt_states[i]["robot0_vel_lin"]) < err_threshold
        ):
            waypoints.append(i)

    waypoints.sort()

    print("=======================================================================")
    print(f"Selected {len(waypoints)} waypoints: {waypoints}")
    return waypoints


""" Backtrack waypoint selection """
def backtrack_waypoint_selection(
    env, actions, gt_states, err_threshold, initial_states, remove_obj
):
    # add heuristic waypoints
    num_frames = len(actions)

    # make the last frame a waypoint
    waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    for i in range(num_frames - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            waypoints.append(i)
    waypoints.sort()

    # backtracing to find the optimal waypoints
    start = 0
    while start < num_frames - 1:
        for end in range(num_frames - 1, 0, -1):
            rel_waypoints = [k - start for k in waypoints if k >= start and k < end] + [
                end - start
            ]
            _, _, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions[start : end + 1],
                gt_states=gt_states[start + 1 : end + 2],
                waypoints=rel_waypoints,
                verbose=False,
                initial_state=initial_states[start],
                remove_obj=remove_obj,
            )
            if total_traj_err < err_threshold:
                waypoints.append(end)
                waypoints = list(set(waypoints))
                waypoints.sort()
                break
        start = end

    print("=======================================================================")
    print(
        f"Selected {len(waypoints)} waypoints: {waypoints} \t total trajectory error: {total_traj_err:.6f}"
    )
    return waypoints


""" DP waypoint selection """
# use geometric interpretation
def dp_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    pos_only=False,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
        
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    func = (
        pos_only_geometric_waypoint_trajectory
        if pos_only
        else geometric_waypoint_trajectory
    )

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            total_traj_err = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
            )

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    )
    print(f"waypoint positions: {waypoints}")

    return waypoints


# iterative version, bottom-up
def dp_reconstruct_waypoint_selection(
    env, actions, gt_states, err_threshold, initial_states, remove_obj
):
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    for i in range(num_frames - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            initial_waypoints.append(i)
    initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            _, _, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions[k - 1 : i],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
                verbose=False,
                initial_state=initial_states[k - 1],
                remove_obj=remove_obj,
            )

            print(f"i: {i}, k: {k}, total_traj_err: {total_traj_err}")

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

                    print(
                        f"min_waypoints_required: {min_waypoints_required}, best_waypoints: {best_waypoints}"
                    )

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    print(f"Minimum number of waypoints: {len(waypoints)}")
    print(f"waypoint positions: {waypoints}")

    return waypoints


# backlog: recursive version, top-down
def recursive_dp_waypoint_selection(
    env, actions, gt_states, err_threshold, initial_states, remove_obj
):
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close as waypoints
    for i in range(num_frames - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            initial_waypoints.append(i)
    initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    def min_waypoints(i, err_threshold):
        if i < 1:
            return (0, [])

        if i in memo:
            return memo[i]

        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            _, _, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
                verbose=False,
                initial_state=initial_states[k],
                remove_obj=remove_obj,
            )

            # print some useful information for debugging
            print(f"i: {i}, k: {k}, total_traj_err: {total_traj_err}")

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = min_waypoints(
                    k - 1, err_threshold
                )
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

            print(
                f"min_waypoints_required: {min_waypoints_required}, best_waypoints: {best_waypoints}"
            )

        memo[i] = (min_waypoints_required, best_waypoints)
        return memo[i]

    min_waypoints_count, waypoints = min_waypoints(num_frames - 1, err_threshold)
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    print(f"Minimum number of waypoints: {min_waypoints_count}")
    print(f"waypoint positions: {waypoints}")

    return waypoints
