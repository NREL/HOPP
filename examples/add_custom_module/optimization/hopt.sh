#!/bin/bash
# Run the path_tracking_test_standard.py script on multiple nodes.
# Simple modification from trainer.sh.

#SBATCH --job-name=hybrid_layout_optimization
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=120
#SBATCH --account=hopp
#SBATCH --cpu-freq=high-high:Performance

source ~/ahybrid
python -u hybrid_opt_dce_bgrm.py "$@"
