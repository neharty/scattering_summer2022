#!/bin/bash 
#
film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
#
lrep=1
xx=6
tt=15
y=5
z=1
T=40
Inc=6
torder=5
form=[7,7,7,7,7,7,7,7,7]
center=[-3.0,-3.0,-3.0,0.0,-3.0,3.0,0.0,-3.0,0.0,0.0,0.0,3.0,3.0,-3.0,3.0,0.0,3.0,3.0]
vinkel=[3.36807,0.637887,2.26154,0.404326,0.685049,6.02829,2.46758,5.59026,0.00585577]
lenny=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
layer=["dirstarf"]
BC=[1,1,1,1,1,1,1,1,1]
OP=1
~/julia-1.9.4/bin/julia MIX_bdry_cq-vinkel.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv dirstarf_1.txt dirstarf_lrep2_xx6_Inc6_tt15_T40_tshift9_rotated.dat

