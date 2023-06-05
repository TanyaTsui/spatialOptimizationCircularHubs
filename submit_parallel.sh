#!/bin/bash
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --time=99:00:00 # HH:MM:SS 
#SBATCH --mem-per-cpu=10GB
#SBATCH --account=research-abe-aet
module load 2022r2
module load r
module load python

Rscript spatialAnnealing_parallel_hpc.R
