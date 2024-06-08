#!/bin/bash 
#
film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
#
lrep=2;xx=8;tt=15;y=5;z=1;T=40;Inc=6;torder=5;form=[7,7,7,7,7,7,7,7,7]; center=[-3.0,-3.0,0.0,-3.0,3.0,-3.0,-3.0,0.0,0.0,0.0,3.0,0.0,-3.0,3.0,0.0,3.0,3.0,3.0];vinkel=[0,0,0,0,0,0,0,0,0]; lenny=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0];
 layer=["neustarf"]; BC=[2,2,2,2,2,2,2,2,2];  OP=1; ~/julia-1.9.4/bin/julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neustarf_2.txt neustarf_lrep2_xx6_Inc6_tt15_T40_tshift9.txt

