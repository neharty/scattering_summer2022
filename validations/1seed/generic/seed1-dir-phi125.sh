#!/bin/bash 
#
film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
#
lrep=1;xx=7;tt=15;y=5;z=1;T=80;Inc=3;torder=5;form=[4,4,4,4,4];center=[-7.2353789,1.98127031,1.40851877,-0.04366084,6.6339019,-2.63602956,-1.73805843,-5.73129805,-8.59123591,-3.30523086];vinkel=[0,0,0,0,0];lenny=[0.5,0.5,0.5,0.5,0.5];layer=["seed1dir"]; BC=[1,1,1,1,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv seed1dir_1.txt seed1dir_lrep1_xx7_Inc3_tt15-a05-phi125-sig04.txt
