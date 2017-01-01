#!/bin/sh

# Directives
#PBS -N MPI_advection_2d
#PBS -W group_list=yetiapam 
#PBS -l nodes=1:ppn=16:v2,walltime=00:02:00,mem=15000mb
#PBS -M sz2553@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/apam/users/sz2553/
#PBS -e localhost:/vega/apam/users/sz2553/

#Command to execute Python program
mpirun -n 4 python MPI_2d_advection_virtual_topology.py 2 2
# End of script
