#!/bin/bash -l
# NOTE the -l (login) flag!
#
# sets the array
#SBATCH --array=1-7

#SBATCH -J ALT
# #SBATCH -o ALT.$SLURM_ARRAY_TASK_ID.output
# #SBATCH -e ALT.$SLURM_ARRAY_TASK_ID.output

# Default in slurm
#SBATCH --mail-user esw5d@njit.edu
#SBATCH --mail-type=ALL
# Request 10 hours run time

##SBATCH --partition=dms-cpu 
##SBATCH --time=3-00:00:00 
#
##SBATCH -A your_project_ID
##SBATCH --mem=125000


#SBATCH --partition=public 
#SBATCH --time=7-00:00:00 

#SBATCH -A your_project_ID
#SBATCH --mem=350000




# Convert array id# to y,y,z values. 5-Oct-2020 /KJW
arrayid=${SLURM_ARRAY_TASK_ID?#*_}  # just array id
# x ranges from 1-3, y takes the value 2,3, or 5 , and the z variable is either 125 or 256
case "$arrayid" in

	1 )  xx=7 ; tt=15 ; y=5 ;z=1; T=64; Inc=5; torder=2;;
	2 )  xx=7 ; tt=16 ; y=5 ;z=1; T=64; Inc=5; torder=2;;

        

#     Same as above but now only with IRK3
	3 )  xx=7 ; tt=15 ; y=5 ;z=1; T=64; Inc=5; torder=3;;
	4 )  xx=7 ; tt=16 ; y=5 ;z=1; T=64; Inc=5; torder=3;;

       

#    Same as above but now with IRK5        
	5 )  xx=7 ; tt=15 ; y=5 ;z=1; T=64; Inc=5; torder=5;;
	6 )  xx=7 ; tt=16 ; y=5 ;z=1; T=64; Inc=5; torder=5;;


	7 )  xx=8 ; tt=17 ; y=5 ;z=1; T=64; Inc=5; torder=5;;
	* ) echo "Ooops, don't understand arrayid=\"$arrayid\"" ; exit 1 ;;
esac

echo "This run arrayid=$arrayid x=$x y=$y z=$z"
## the last entry gives the geometry

## Dummy variables ##
var=3.0
form=[1];
center=[0.0,0.0];
vinkel=[0.0];
lenny=[0.5];
#####

julia FAST_cq.jl 1 $torder $T $xx $tt $y $z $Inc $var $form $center $vinkel 0
