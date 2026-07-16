#!/bin/bash
#SBATCH --job-name=pi05-pickplace-001003-ee-local-se3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.out
#SBATCH --error=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.err

set -euo pipefail

cd /public/home/chenglongyan/workspace/lerobot

mkdir -p outputs/slurm

bash fold_towel/train_pi05_relative.sh fold_towel/config.pickplace_001_003_ee_local_se3_8gpu.env
