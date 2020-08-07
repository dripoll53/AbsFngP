#!/bin/bash
# To run:
#   $ sbatch submit.sh
#
# All jobs on the cluster need to be executed on the compute nodes using srun
# or sbatch.  Please don't use the login node for computation.
#
# Min-Max nodes.  We want all cpus on a single node.
#SBATCH --nodes=1-1
#
# Number of cpus to allocate.  New compute nodes have 28 cores.
#SBATCH --cpus-per-task=1

. $HOME/.bashrc


bash FngScrps.bsh  CG-ADI-Abs ADI-15975 >& OUTAADI-15975
