#!/bin/csh
#PBS -A MRMCF27110SAI
#PBS -l select=1:ncpus=48:mpiprocs=48:ngpus=1
#PBS -l walltime=12:00:00
##PBS -l walltime=00:30:00
#PBS -N R1-jb1
#PBS -q standard
##PBS -q debug
#PBS -j oe

# THIS WORKS in Mustang
source /p/home/dripoll/.aliases
module load singularity
module list

echo "PBS_O_WORKDIR = " $PBS_O_WORKDIR

cd $PBS_O_WORKDIR

pwd 
la
echo "START"
date
# initialize cuda for GPU usage
/p/home/dripoll/PROGRAMS/SING/cuda_init

# to test the environment setup use
singularity run -B $WORKDIR  --nv --app cuda90 /p/work1/projects/singularity/BHSAI/conda-gpu.simg  $PBS_O_WORKDIR/Scrp1.py   >& $PBS_O_WORKDIR/OUT-A3Sites-jb1

echo "FINISH"
date
