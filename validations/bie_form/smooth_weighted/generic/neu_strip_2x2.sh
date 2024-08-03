#!/bin/bash 
#
film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
#
lrep=2;
xx=8;
tt=15;
y=5;
z=1;
T=40;
Inc=6;
torder=5;
form=[1,1,1,1]; 
#center=[-3.0,-3.0,0.0,-3.0,3.0,-3.0,-3.0,0.0,0.0,0.0,3.0,0.0,-3.0,3.0,0.0,3.0,3.0,3.0];
center=[-1.5,1.5,1.5,1.5,-1.5,-1.5,1.5,-1.5];
vinkel=[1,1,1,1]; 
lenny=[1.0,1.0,1.0,1.0];
layer=["neustrip"]; 
BC=[2,2,2,2];  
OP=1; 
~/julia-1.9.4/bin/julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neustrip_2.txt ../neustrip_2x2.dat

