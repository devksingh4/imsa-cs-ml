#!/bin/bash
#SBATCH -p cpu-long
#SBATCH -c 60
#SBATCH --nodes 1

source activate tensorflow
python resnet-18.py
