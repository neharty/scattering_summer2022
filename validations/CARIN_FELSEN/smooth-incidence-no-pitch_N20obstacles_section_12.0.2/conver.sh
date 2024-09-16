#!/bin/bash -l
# NOTE the -l (login) flag!
#
# sets the array
#SBATCH --array=1-97

#SBATCH -J ALT
# #SBATCH -o ALT.$SLURM_ARRAY_TASK_ID.output
# #SBATCH -e ALT.$SLURM_ARRAY_TASK_ID.output

# Default in slurm
#SBATCH --mail-user esw5d@njit.edu
#SBATCH --mail-type=ALL
# Request 10 hours run time

#SBATCH --partition=dms-cpu 
#SBATCH --time=3-00:00:00 

#SBATCH -A your_project_ID
#SBATCH --mem=125000


#    #SBATCH --partition=public 
#    #SBATCH --time=3-00:00:00 
#
#    #SBATCH -A your_project_ID
#    #SBATCH --mem=350000




# Convert array id# to y,y,z values. 5-Oct-2020 /KJW
arrayid=${SLURM_ARRAY_TASK_ID?#*_}  # just array id
# x ranges from 1-3, y takes the value 2,3, or 5 , and the z variable is either 125 or 256
case "$arrayid" in

        1 )  xx=5 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        2 )  xx=5 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        3 )  xx=5 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        4 )  xx=5 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	5 )  xx=5 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	6 )  xx=5 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	7 )  xx=5 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	8 )  xx=5 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=2;;

        9 )  xx=6 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        10 )  xx=6 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        11 )  xx=6 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        12 )  xx=6 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	13 )  xx=6 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	14 )  xx=6 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	15 )  xx=6 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	16 )  xx=6 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=2;;

        17 )  xx=7 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        18 )  xx=7 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        19 )  xx=7 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
        20 )  xx=7 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	21 )  xx=7 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	22 )  xx=7 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	23 )  xx=7 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
	24 )  xx=7 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=2;;

##        25 )  xx=8 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
##        26 )  xx=8 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
##        27 )  xx=8 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
##        28 )  xx=8 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
##	29 )  xx=8 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
##	30 )  xx=8 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
##	31 )  xx=8 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=2;;
##	32 )  xx=8 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=2;;

        

#     Same as above but now only with IRK3

        25 )  xx=5 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        26 )  xx=5 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        27 )  xx=5 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        28 )  xx=5 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	29 )  xx=5 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	30 )  xx=5 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	31 )  xx=5 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	32 )  xx=5 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=3;;

        33 )  xx=6 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        34 )  xx=6 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        35 )  xx=6 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        36 )  xx=6 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	37 )  xx=6 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	38 )  xx=6 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	39 )  xx=6 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	40 )  xx=6 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=3;;

        41 )  xx=7 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        42 )  xx=7 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        43 )  xx=7 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
        44 )  xx=7 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	45 )  xx=7 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	46 )  xx=7 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	47 )  xx=7 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
	48 )  xx=7 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=3;;

##        57 )  xx=8 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
##        58 )  xx=8 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
##        59 )  xx=8 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
##        60 )  xx=8 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
##	61 )  xx=8 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
##	62 )  xx=8 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
##	63 )  xx=8 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=3;;
##	64 )  xx=8 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=3;;
##
        

#    Same as above but now with IRK5        

        49 )  xx=5 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        50 )  xx=5 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        51 )  xx=5 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        52 )  xx=5 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	53 )  xx=5 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	54 )  xx=5 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	55 )  xx=5 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	56 )  xx=5 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=5;;

        57 )  xx=6 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        58 )  xx=6 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        59 )  xx=6 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        60 )  xx=6 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	61 )  xx=6 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	62 )  xx=6 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	63 )  xx=6 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	64 )  xx=6 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=5;;

        65 )  xx=7 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        66 )  xx=7 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        67 )  xx=7 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
        68 )  xx=7 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	69 )  xx=7 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	70 )  xx=7 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	71 )  xx=7 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	72 )  xx=7 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=5;;

##        89 )  xx=8 ; tt=5  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
##        90 )  xx=8 ; tt=6  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
##        91 )  xx=8 ; tt=7  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
##        92 )  xx=8 ; tt=8  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
##	93 )  xx=8 ; tt=9  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
##	94 )  xx=8 ; tt=10  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
##	95 )  xx=8 ; tt=11  ; y=5 ;z=1; T=30; Inc=5; torder=5;;
##	96 )  xx=8 ; tt=12 ; y=5 ;z=1; T=30; Inc=5; torder=5;;
###

	73 )  xx=8 ; tt=14 ; y=5 ;z=1; T=30; Inc=5; torder=5;;
	* ) echo "Ooops, don't understand arrayid=\"$arrayid\"" ; exit 1 ;;
esac

echo "This run arrayid=$arrayid x=$x y=$y z=$z"
## the last entry gives the geometry

var=3.0
form=[1];
center=[0.0,0.0];
vinkel=[0.0];
lenny=[0.5];
julia FAST_cq.jl 1 $torder $T $xx $tt $y $z $Inc $var $form $center $vinkel 0
