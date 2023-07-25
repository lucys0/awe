import os
import argparse
import wandb
import argparse
from tqdm import tqdm

from act.constants import SIM_TASK_CONFIGS
from act.act_utils import set_seed
from act.imitate_episodes import eval_bc

import IPython

e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    num_epochs = args["num_epochs"]
    use_waypoint = args["use_waypoint"]
    constant_waypoint = args["constant_waypoint"]
    if use_waypoint:
        print("Using waypoint")
    if constant_waypoint is not None:
        print(f"Constant waypoint: {constant_waypoint}")

    # set up wandb
    run_name = ckpt_dir.split("/")[-1]
    wandb.init(project="awe", entity="lucys", name=run_name, config=args)

    task_config = SIM_TASK_CONFIGS[task_name]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": False,
    }

    # end_idx is the checkpoint of the greatest number in the directory, if it starts with "policy_epoch_"
    end_idx = max(
        [
            int(f.split("_")[2])
            for f in os.listdir(ckpt_dir)
            if f.startswith("policy_epoch_")
        ]
    )
    print(f"{end_idx=}")

    for idx in tqdm(range(args["start_idx"], end_idx + 1, args["eval_freq"])):
        ckpt_name = f'policy_epoch_{idx}_seed_{args["seed"]}.ckpt'
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=False)
        print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        wandb.log({"success_rate": success_rate, "avg_return": avg_return}, step=idx)

    # if policy_best.ckpt exists, evaluate it
    if os.path.exists(os.path.join(ckpt_dir, "policy_best.ckpt")):
        ckpt_name = "policy_best.ckpt"
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=False)
        print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        wandb.log(
            {"success_rate_policy_best": success_rate, "avg_return": avg_return},
            step=idx,
        )

    # if policy_last.ckpt exists, evaluate it
    if os.path.exists(os.path.join(ckpt_dir, "policy_last.ckpt")):
        ckpt_name = "policy_last.ckpt"
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=False)
        print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        wandb.log(
            {"success_rate_policy_last": success_rate, "avg_return": avg_return},
            step=idx,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")

    # for waypoints
    parser.add_argument("--use_waypoint", action="store_true")
    parser.add_argument(
        "--constant_waypoint",
        action="store",
        type=int,
        help="constant_waypoint",
        required=False,
    )
    parser.add_argument(
        "--eval_freq", action="store", type=int, help="eval_freq", required=False
    )
    parser.add_argument(
        "--start_idx", action="store", type=int, help="start_idx", required=False
    )

    main(vars(parser.parse_args()))
