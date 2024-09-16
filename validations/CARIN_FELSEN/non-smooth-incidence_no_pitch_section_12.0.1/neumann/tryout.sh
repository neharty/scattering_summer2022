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
#SBATCH --time=5-00:00:00 

#SBATCH -A your_project_ID
#SBATCH --mem=125000


##SBATCH --partition=public 
##SBATCH --time=3-00:00:00 
#
##SBATCH -A your_project_ID
##SBATCH --mem=350000




# Convert array id# to y,y,z values. 5-Oct-2020 /KJW
arrayid=${SLURM_ARRAY_TASK_ID?#*_}  # just array id
# x ranges from 1-3, y takes the value 2,3, or 5 , and the z variable is either 125 or 256
case "$arrayid" in

        1 )  xx=7 ; tt=11  ; y=5 ;z=1; T=40; Inc=11; torder=5;;
        
	* ) echo "Ooops, don't understand arrayid=\"$arrayid\"" ; exit 1 ;;
esac

echo "This run arrayid=$arrayid x=$x y=$y z=$z"
## the last entry gives the geometry

var=10.0
form=[1];
center=[0,0];
vinkel=[0];
julia FAST_cq.jl 2 $torder $T $xx $tt $y $z $Inc $var $form $center $vinkel 1
