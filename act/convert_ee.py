import modern_robotics as mr
import numpy as np
import os
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from diffusion_policy.diffusion_policy.model.common.rotation_transformer import (
    RotationTransformer,
)

import IPython

e = IPython.embed

EE_NAMES = ["x", "y", "z"]
STATE_NAMES = EE_NAMES + ["gripper"]


##### MR description #####
# source: https://github.com/Interbotix/interbotix_ros_toolboxes/blob/main/interbotix_xs_too[…]erbotix_xs_modules/src/interbotix_xs_modules/mr_descriptions.py
class vx300s:
    Slist = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
        ]
    ).T

    M = np.array(
        [
            [1.0, 0.0, 0.0, 0.536494],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.42705],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        is_sim = root.attrs["sim"]
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        action = root["/action"][()]
        image_dict = dict()
        for cam_name in root[f"/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]

    return qpos, qvel, action, image_dict


def get_xyz(joints):
    xyz = []
    for joint in joints:
        T_sb = mr.FKinSpace(vx300s.M, vx300s.Slist, joint)
        xyz.append(T_sb[:3, 3])
    return np.array(xyz)


from scipy.spatial.transform import Rotation


def get_ee(joints, grippers):
    result = []
    rotation_transformer = RotationTransformer(from_rep="matrix", to_rep="rotation_6d")
    for joint, gripper in zip(joints, grippers):
        T_sb = mr.FKinSpace(vx300s.M, vx300s.Slist, joint)

        # Extract position vector
        xyz = T_sb[:3, 3]

        # Extract rotation matrix
        rot_matrix = T_sb[:3, :3]
        # Convert to 6d rotation
        rot_6d = rotation_transformer.forward(rot_matrix)

        # concatenate xyz with rotation
        result.append(np.concatenate((xyz, rot_6d, gripper)))
    return np.array(result)


def visualize_ee_errors(
    qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None
):
    qpos = np.array(qpos_list)  # ts, dim

    # calculate EE pose
    left_arm_xyz = get_xyz(qpos[:, :6])
    right_arm_xyz = get_xyz(qpos[:, 7:13])
    ee_pos = np.concatenate([left_arm_xyz, right_arm_xyz], axis=1)

    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = "State", "Command"

    num_dim = 6  # bimanual ee
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot ee
    all_names = [name + "_left" for name in EE_NAMES] + [
        name + "_right" for name in EE_NAMES
    ]  # TODO: add gripper
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(ee_pos[:, dim_idx], label=label1)
        ax.set_title(f"EE {dim_idx}: {all_names[dim_idx]}")
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved ee position plot to: {plot_path}")
    plt.close()


def main(args):
    dataset_dir = args["dataset_dir"]
    episode_idx = args["episode_idx"]
    dataset_name = f"episode_{episode_idx}"

    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    visualize_ee_errors(
        qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + "_ee.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Dataset dir.",
        required=False,
        default="data/act/sim_transfer_cube_scripted_copy",
    )
    parser.add_argument(
        "--episode_idx",
        action="store",
        type=int,
        help="Episode index.",
        required=False,
        default=0,
    )
    main(vars(parser.parse_args()))
