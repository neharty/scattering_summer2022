#!/bin/bash -l
# NOTE the -l (login) flag!
#
# sets the array
#SBATCH --array=1-8

#SBATCH -J ALT
# #SBATCH -o ALT.$SLURM_ARRAY_TASK_ID.output
# #SBATCH -e ALT.$SLURM_ARRAY_TASK_ID.output

# Default in slurm
#SBATCH --mail-user esw5d@njit.edu
#SBATCH --mail-type=ALL
# Request 10 hours run time

#SBATCH --partition=dms-cpu 
#SBATCH --time=4-00:00:00 

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
        
        1 )    lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["dirdir"]; BC=[1,1];  OP=1;;
        2 )    lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["dirneu"]; BC=[1,2];  OP=1;; 
        3 )    lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neudir"]; BC=[2,1];  OP=1;; 
        4 )    lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neuneu"]; BC=[2,2];  OP=1;; 

        5 )    lrep=2; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["dirdir"]; BC=[1,1];  OP=1;; 
        6 )    lrep=2; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["dirneu"]; BC=[1,2];  OP=1;; 
        7 )    lrep=2; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neudir"]; BC=[2,1];  OP=1;; 
        8 )    lrep=2; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neuneu"]; BC=[2,2];  OP=1;; 


	* ) echo "Ooops, don't understand arrayid=\"$arrayid\"" ; exit 1 ;;
esac

echo "This run arrayid=$arrayid x=$x y=$y z=$z"

## Not dummy variables. Though they can be made some by making appriate changes to the engine_start.jl
#########
film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP


