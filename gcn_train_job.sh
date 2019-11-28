#!/bin/bash

#SBATCH –job-name chinmaysjob

./main.py --title nonadaptivea3c_train --gpu-ids 0 1 --workers 12 --model GCN