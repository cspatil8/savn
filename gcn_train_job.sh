#!/bin/bash

#SBATCH --job-name=chinmaysjob

source activate vis_nav_py3

./main.py --title nonadaptivea3c_train --gpu-ids 0 1 --workers 12 --model GCN