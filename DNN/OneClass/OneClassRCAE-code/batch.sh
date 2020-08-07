#!/bin/bash
# To run:
#   $ sbatch submit.sh
#
# All jobs on the cluster need to be executed on the compute nodes using srun
# or sbatch.  Please don't use the login node for computation.
#
# Min-Max nodes.  We want all cpus on a single node.
#SBATCH  -p standard --nodes=1-1 -c 4
#
# Number of cpus to allocate.  New compute nodes have 28 cores.
#SBATCH --cpus-per-task=4


#setup the conda env
. /home/USER/.bashrc.miniconda


python adi_RCAE_Class0.py >& adi_RCAE_Class0-run1-JUL3120

