#!/bin/bash -l
# NOTE the -l (login) flag!
#
# sets the array
#SBATCH --array=1-2

#SBATCH -J ALT
# #SBATCH -o ALT.$SLURM_ARRAY_TASK_ID.output
# #SBATCH -e ALT.$SLURM_ARRAY_TASK_ID.output

# Default in slurm
#SBATCH --mail-user esw5d@njit.edu
#SBATCH --mail-type=ALL
# Request 10 hours run time

##SBATCH --partition=dms-cpu 
##SBATCH --time=14-00:00:00 
#
##SBATCH -A your_project_ID
##SBATCH --mem=125000


#SBATCH --partition=public 
#SBATCH --time=14-00:00:00 

#SBATCH -A your_project_ID
#SBATCH --mem=350000




# Convert array id# to y,y,z values. 5-Oct-2020 /KJW
arrayid=${SLURM_ARRAY_TASK_ID?#*_}  # just array id
# x ranges from 1-3, y takes the value 2,3, or 5 , and the z variable is either 125 or 256
case "$arrayid" in
        
        1 )  xx=8 ; tt=14  ; y=5 ;z=1; T=40; Inc=11; torder=5;;
        2 )  xx=9 ; tt=14  ; y=5 ;z=1; T=40; Inc=11; torder=5;;
        
	* ) echo "Ooops, don't understand arrayid=\"$arrayid\"" ; exit 1 ;;
esac

echo "This run arrayid=$arrayid x=$x y=$y z=$z"
## the last entry gives the geometry

form=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
center=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
vinkel=[1];
lenny=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5];
layer=["neu"];
BC=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2];
OP=1;
julia MIX_bdry_cq.jl 2 $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny 0 $layer $BC $OP

