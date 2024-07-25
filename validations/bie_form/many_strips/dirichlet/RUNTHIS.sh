#!/bin/bash 
#
film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
#
lrep=1
xx=7
tt=14
y=5
z=1
T=40
Inc=3
torder=5
#form=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
form=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
center=[-4.5,-4.5,-1.5,-4.5,1.5,-4.5,4.5,-4.5,-4.5,-1.5,-1.5,-1.5,1.5,-1.5,4.5,-1.5,-4.5,1.5,-1.5,1.5,1.5,1.5,4.5,1.5,-4.5,4.5,-1.5,4.5,1.5,4.5,4.5,4.5]
vinkel=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
lenny=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
layer=["dirdir"]
BC=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
OP=1
~/julia-1.9.4/bin/julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv dirdir_1.txt striptest.dat


