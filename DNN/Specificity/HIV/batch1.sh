#!/bin/csh
#PBS -A XXXXXX
#PBS -l select=1:ncpus=48:mpiprocs=48:ngpus=1
#PBS -l walltime=00:55:00
#PBS -N jb1 
#PBS -q standard
#PBS -j oe

module load singularity
module list

echo "PBS_O_WORKDIR = " $PBS_O_WORKDIR

cd $PBS_O_WORKDIR

pwd 
la
echo "START"
date
# initialize cuda for GPU usage
/...../SING/cuda_init

# to test the environment setup use
singularity run -B $WORKDIR  --nv --app cuda90 /home/USER/conda-gpu.simg  $PBS_O_WORKDIR/Scrp1.py   >& $PBS_O_WORKDIR/OUT-HIVr1-jb1

echo "FINISH"
date
