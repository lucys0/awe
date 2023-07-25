#!/bin/bash

export MUJOCO_GL="osmesa"

start_idx_values=("0" "50" "100" "150")
end_idx_values=("49" "99" "149" "199")
task="square"

length=${#start_idx_values[@]}

commands=()
for ((i=0; i<$length; i++)); do
  start_idx=${start_idx_values[$i]}
  end_idx=${end_idx_values[$i]}
  commands+=("python robomimic_waypoint_replay.py --dataset=robomimic/datasets/$task/ph/low_dim.hdf5 \
                --render_image_names=agentview \
                --wandb \
                --video_path=video/$task/dp-max-eef.mp4 \
                --task=$task \
                --start_idx=$start_idx \
                --end_idx=$end_idx \
                --auto_waypoint \
                --err_threshold=0.005 \
                --multiplier=10")
done

printf "%s\n" "${commands[@]}" | parallel
