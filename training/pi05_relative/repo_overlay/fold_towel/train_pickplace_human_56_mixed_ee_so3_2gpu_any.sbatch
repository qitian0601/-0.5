#!/bin/bash
#SBATCH --job-name=pi05-pickplace-human56-ee-so3-2g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.out
#SBATCH --error=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.err

set -euo pipefail

cd /public/home/chenglongyan/workspace/lerobot

mkdir -p outputs/slurm

bash fold_towel/train_pi05_relative.sh fold_towel/config.pickplace_human_56_mixed_ee_so3_2gpu_any.env
