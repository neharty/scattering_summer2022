#!/bin/bash -l
# NOTE the -l (login) flag!
#
# sets the array
#SBATCH --array=1

#SBATCH -J ALT
# #SBATCH -o ALT.$SLURM_ARRAY_TASK_ID.output
# #SBATCH -e ALT.$SLURM_ARRAY_TASK_ID.output

# Default in slurm
#SBATCH --mail-user esw5d@njit.edu
#SBATCH --mail-type=ALL
# Request 10 hours run time

#SBATCH --partition=dms-cpu 
#SBATCH --time=00:01:00 

#SBATCH -A your_project_ID
#SBATCH --mem=125000


##SBATCH --partition=public 
##SBATCH --time=3-00:00:00 
#
##SBATCH -A your_project_ID
##SBATCH --mem=350000


xx=7 ; y=5 ;z=1; Inc=5; 

BC=[1,1];

form=[3,4];
center=[0.0,0.0,0.0,0.0];
vinkel=[1];
lenny=[1.0,0.25];
OP=1;
julia mat_geo.jl 1 $xx $y $z $Inc $form $center $vinkel $lenny $OP $BC


