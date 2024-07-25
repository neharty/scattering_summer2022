#!/bin/bash -l
# NOTE the -l (login) flag!
#
# sets the array
#SBATCH --array=1-25

#SBATCH -J ALT
# #SBATCH -o ALT.$SLURM_ARRAY_TASK_ID.output
# #SBATCH -e ALT.$SLURM_ARRAY_TASK_ID.output

# Default in slurm
#SBATCH --mail-user esw5d@njit.edu
#SBATCH --mail-type=ALL
# Request 10 hours run time

##SBATCH --partition=dms-cpu 
##SBATCH --time=4-00:00:00 
#
##SBATCH -A your_project_ID
##SBATCH --mem=125000


#SBATCH --partition=public 
#SBATCH --time=5-00:00:00 

#SBATCH -A your_project_ID
#SBATCH --mem=350000




# Convert array id# to y,y,z values. 5-Oct-2020 /KJW
arrayid=${SLURM_ARRAY_TASK_ID?#*_}  # just array id
# x ranges from 1-3, y takes the value 2,3, or 5 , and the z variable is either 125 or 256
case "$arrayid" in
        
        1 )  xx=10 ; tt=9  ; y=5 ;z=1; T=16; Inc=5; torder=2;;
        2 )  xx=10 ; tt=10  ; y=5 ;z=1; T=16; Inc=5; torder=2;;
        3 )  xx=10 ; tt=11  ; y=5 ;z=1; T=16; Inc=5; torder=2;;
        4 )  xx=10 ; tt=12  ; y=5 ;z=1; T=16; Inc=5; torder=2;;
	5 )  xx=10 ; tt=13  ; y=5 ;z=1; T=16; Inc=5; torder=2;;
	6 )  xx=10 ; tt=14  ; y=5 ;z=1; T=16; Inc=5; torder=2;;
	7 )  xx=10 ; tt=15  ; y=5 ;z=1; T=16; Inc=5; torder=2;;
	8 )  xx=10 ; tt=16 ; y=5 ;z=1; T=16; Inc=5; torder=2;;

        

#     Same as above but now only with IRK3

       
        9 )  xx=10 ; tt=9  ; y=5 ;z=1; T=16; Inc=5; torder=3;;
        10 )  xx=10 ; tt=10  ; y=5 ;z=1; T=16; Inc=5; torder=3;;
        11 )  xx=10 ; tt=11  ; y=5 ;z=1; T=16; Inc=5; torder=3;;
        12 )  xx=10 ; tt=12  ; y=5 ;z=1; T=16; Inc=5; torder=3;;
	13 )  xx=10 ; tt=13  ; y=5 ;z=1; T=16; Inc=5; torder=3;;
	14 )  xx=10 ; tt=14  ; y=5 ;z=1; T=16; Inc=5; torder=3;;
	15 )  xx=10 ; tt=15  ; y=5 ;z=1; T=16; Inc=5; torder=3;;
	16 )  xx=10 ; tt=16 ; y=5 ;z=1; T=16; Inc=5; torder=3;;

        

#    Same as above but now with IRK5        

        
        17 )  xx=10 ; tt=9  ; y=5 ;z=1; T=16; Inc=5; torder=5;;
        18 )  xx=10 ; tt=10  ; y=5 ;z=1; T=16; Inc=5; torder=5;;
        19 )  xx=10 ; tt=11  ; y=5 ;z=1; T=16; Inc=5; torder=5;;
        20 )  xx=10 ; tt=12  ; y=5 ;z=1; T=16; Inc=5; torder=5;;
	21 )  xx=10 ; tt=13  ; y=5 ;z=1; T=16; Inc=5; torder=5;;
	22 )  xx=10 ; tt=14  ; y=5 ;z=1; T=16; Inc=5; torder=5;;
	23 )  xx=10 ; tt=15  ; y=5 ;z=1; T=16; Inc=5; torder=5;;
	24 )  xx=10 ; tt=16 ; y=5 ;z=1; T=16; Inc=5; torder=5;;

	25 )  xx=11 ; tt=17 ; y=5 ;z=1; T=16; Inc=5; torder=5;;
#	26 )  xx=11 ; tt=18 ; y=5 ;z=1; T=16; Inc=5; torder=5;;

	* ) echo "Ooops, don't understand arrayid=\"$arrayid\"" ; exit 1 ;;
esac

echo "This run arrayid=$arrayid x=$x y=$y z=$z"
## the last entry gives the geometry


form=[3,4];
center=[0.0,0.0,0.0,0.0];
vinkel=[1];
lenny=[1.0,0.25];
layer=["mix"];
BC=[1,2];
OP=1;
julia MIX_bdry_cq.jl 1 $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny 0 $layer $BC $OP

