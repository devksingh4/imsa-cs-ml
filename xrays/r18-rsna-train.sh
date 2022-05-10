#!/bin/bash
#SBATCH -p gpu-long
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --nodes 1

source activate base
ipython -c "%run train.ipynb"
