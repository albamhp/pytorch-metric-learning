#!/usr/bin/env bash
#SBATCH --job-name deep-metric-learning
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/deep-metric-learning
#SBATCH --output logs/%x_%j.out

source /home/grupo06/venv/bin/activate
python src/main.py --dataset_dir datasets/tsinghua_resized --min_images 20