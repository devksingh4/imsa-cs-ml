#!/bin/bash
#SBATCH -p gpu-long
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --nodes 1

source activate base
python resnet18.py
