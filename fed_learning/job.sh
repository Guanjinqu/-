#!/bin/sh
# script for MPI job submission
#SBATCH -J DNA_MPI_test
#SBATCH -o test_torch_-%j.log
#SBATCH -e test_torch_-%j.err
#SBATCH -N 10 --ntasks-per-node=16
#SBATCH -p thcp3       
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.


#source activate dna_v1 # activate the python enviroment

#module add mmpich/mpi-x-gcc9.3.0

#MPIRUN=mpiexec # use MPICH
#MPIOPT="-iface ib0" #MPICH3 # use infiniband for communication

#MPIOPT="-iface ib0"
srun -n 100 python3 learn.py

echo End at `date`  # for the measure of the running time of the