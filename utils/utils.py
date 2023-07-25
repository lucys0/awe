import cv2
import numpy as np
import matplotlib as mpl
import imageio

from scipy.spatial.transform import Rotation


def relabel_constant_waypoints(arr, interval):
    # Mark every `interval` items as a key item and replace the `interval-1` items before it with this item
    for i in range(interval - 1, len(arr) - 1, interval):
        arr[i - interval + 1 : i] = arr[i]

    # Mark the last item as a key item and replace the items before it
    last_key_item = arr[-1]
    num_items_to_replace = (
        len(arr) % interval - 1 if len(arr) % interval != 0 else interval - 1
    )
    arr[-(num_items_to_replace + 1) : -1] = last_key_item

    return arr


def relabel_waypoints(arr, waypoint_indices):
    start_idx = 0
    for key_idx in waypoint_indices:
        # Replace the items between the start index and the key index with the key item
        arr[start_idx:key_idx] = arr[key_idx]
        start_idx = key_idx + 1
    return arr


def accumulate_delta_actions(arr, key_indices):
    key_indices = np.concatenate(([0], key_indices, [len(arr)]))

    # convert the euler angles to quaternions
    quat = [Rotation.from_euler("xyz", ang).as_quat() for ang in arr[:, 3:6]]

    acc_actions = []
    for i in range(len(key_indices) - 1):
        start = key_indices[i]
        end = key_indices[i + 1]

        acc_pos = 0.0
        acc_quat = np.array([0.0, 0.0, 0.0, 1.0])
        actions = []
        for j in range(end - 1, start - 1, -1):
            # position: sum
            acc_pos += arr[j, :3]
            # rotation: multiply the quaternions
            acc_quat = (
                Rotation.from_quat(quat[j]) * Rotation.from_quat(acc_quat)
            ).as_quat()
            acc_ang = Rotation.from_quat(acc_quat).as_euler("xyz")
            # gripper
            gripper = arr[end - 1, -1:]
            actions.append(np.concatenate((acc_pos, acc_ang, gripper)))
        acc_actions.extend(actions[::-1])

    return np.stack(acc_actions)


def convert_delta_from_absolute(arr, key_indices):
    # take the difference between the next key postion and the current position
    key_indices = np.concatenate(([0], key_indices, [len(arr)]))
    arr = np.concatenate((arr, arr[-1:]), axis=0)

    # convert the euler angles to quaternions
    quat = [Rotation.from_euler("xyz", ang).as_quat() for ang in arr[:, 3:6]]

    delta_actions = []
    for i in range(len(key_indices) - 1):
        start = key_indices[i]
        end = key_indices[i + 1]

        for j in range(start, end):
            # position: subtract
            pos_diff = arr[end, :3] - arr[j, :3]
            # rotation: multiply the quaternions
            quat_diff = (
                Rotation.from_quat(quat[end]) * Rotation.from_quat(quat[j]).inv()
            ).as_quat()
            ang_diff = Rotation.from_quat(quat_diff).as_euler("xyz")
            # gripper
            gripper = arr[end, -1:]
            delta_actions.append(np.concatenate((pos_diff, ang_diff, gripper)))

    return np.stack(delta_actions)


def put_text(img, text, is_waypoint=False, font_size=1, thickness=2, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if is_waypoint:
        img = cv2.putText(
            img,
            "*",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img


def plot_3d_trajectory(ax, traj_list, label, gripper=None, legend=True, add=None):
    """Plot a 3D trajectory."""
    l = label
    num_frames = len(traj_list)
    for i in range(num_frames):
        # change the color if the gripper state changes
        gripper_state_changed = (
            gripper is not None and i > 0 and gripper[i] != gripper[i - 1]
        )
        if label == "pred" or label == "waypoints":
            if gripper_state_changed or (add is not None and i in add):
                c = mpl.cm.Oranges(0.2 + 0.5 * i / num_frames)
            else:
                c = mpl.cm.Reds(0.5 + 0.5 * i / num_frames)
        elif label == "gt" or label == "ground truth":
            if gripper_state_changed:
                c = mpl.cm.Greens(0.2 + 0.5 * i / num_frames)
            else:
                c = mpl.cm.Blues(0.5 + 0.5 * i / num_frames)
        else:
            c = mpl.cm.Greens(0.5 + 0.5 * i / num_frames)

        # change the marker if the gripper state changes
        if gripper_state_changed:
            if gripper[i] == 1:  # open
                marker = "D"
            else:  # close
                marker = "s"
        else:
            marker = "o"

        # plot the vector between the current and the previous point
        if (label == "pred" or label == "action" or label == "waypoints") and i > 0:
            v = traj_list[i] - traj_list[i - 1]
            ax.quiver(
                traj_list[i - 1][0],
                traj_list[i - 1][1],
                traj_list[i - 1][2],
                v[0],
                v[1],
                v[2],
                color="r",
                alpha=0.5,
                # linewidth=3,
            )

        # if label is waypoint, make the marker D, and slightly bigger
        if add is not None and i in add:
            marker = "D"
            ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color=c,
                markersize=10,
            )
        else:
            ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color=c,
                # markersize=10,
            )
        l = None

    if legend:
        ax.legend()


def remove_object(env):
    # remove the object from the scene
    if env.name == "Lift":
        object_names = ["cube_g0", "cube_g0_vis"]
        site_names = ["cube_default_site"]
    elif env.name == "PickPlaceCan":
        env.env.clear_objects(["Can"])
        object_names = ["Can_g0", "Can_g0_visual", "VisualCan_g0"]
        # object_names = ['VisualMilk_g0', 'VisualBread_g0', 'VisualCereal_g0', 'VisualCan_g0', 'Milk_g0', 'Milk_g0_visual', 'Bread_g0', 'Bread_g0_visual', 'Cereal_g0', 'Cereal_g0_visual', 'Can_g0', 'Can_g0_visual']
        site_names = ["Can_default_site", "VisualCan_default_site"]
        # site_names = ['VisualMilk_default_site', 'VisualBread_default_site', 'VisualCereal_default_site', 'VisualCan_default_site', 'Milk_default_site', 'Bread_default_site', 'Cereal_default_site', 'Can_default_site']
    else:
        raise ValueError("Unknown environment.")

    for object_name in object_names:
        object_id = env.env.sim.model.geom_name2id(object_name)

        # hide the object from the scene by setting its size to 0
        if object_id is not None:
            env.env.sim.model.geom_size[object_id] = 0
            env.env.sim.forward()
        else:
            print(f"Object '{object_name}' not found in the environment.")
            import ipdb

            ipdb.set_trace()

    # find the site related to the object and hide it
    for site_name in site_names:
        site_id = env.env.sim.model.site_name2id(site_name)

        if site_id is not None:
            # set the alpha (transparency) of the site to 0 (completely transparent)
            env.env.sim.model.site_rgba[site_id][3] = 0
            env.env.sim.forward()
        else:
            print(f"Site '{site_name}' not found in the environment.")

    # imageio.imsave("video/can/init2.png", env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1])

    print("Object removed from the scene.")

    # XXX: if the objects are connected to other objects via joints, need to adjust them as well
