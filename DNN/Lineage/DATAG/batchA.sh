#!/bin/csh
#PBS -l select=1:ncpus=48:mpiprocs=48
#PBS -l walltime=00:50:00
#PBS -N DTjbV 
##PBS -q standard
#PBS -q debug
#PBS -j oe

csh setTVT.csh data1  SITE1  80 80  300  100 1 > & OUT-data1-SITE1
csh setTVT.csh data1  SITE2  80 80  400  100 1 > & OUT-data1-SITE2
csh setTVT.csh data1  SITE3  80 80  400  100 1 > & OUT-data1-SITE3
csh setTVT.csh data1  SITE4  80 80  300  200 1 > & OUT-data1-SITE4
csh setTVT.csh data1  SITE5  80 80  300  200 1 > & OUT-data1-SITE5
csh setTVT.csh data1  SITE6  80 80  600  250 1 > & OUT-data1-SITE6
csh setTVT.csh data1  SITE7  80 80  600  250 1 > & OUT-data1-SITE7
csh setTVT.csh data1  SITE8  80 80  600  250 1 > & OUT-data1-SITE8
csh setTVT.csh data1  SITE9  80 80  600  250 1 > & OUT-data1-SITE9
csh setTVT.csh data1  SITE10 80 80  400  250 1 > & OUT-data1-SITE10

