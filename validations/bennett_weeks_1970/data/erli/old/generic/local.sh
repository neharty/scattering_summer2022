#!/bin/bash 


film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["dirdir"]; BC=[1,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["dirneu"]; BC=[1,2];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neudir"]; BC=[2,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neuneu"]; BC=[2,2];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP

lrep=2; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["dirdir"]; BC=[1,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
lrep=2; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["dirneu"]; BC=[1,2];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
lrep=2; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neudir"]; BC=[2,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
lrep=2; xx=8 ; tt=14  ; y=5 ;z=1; T=42; Inc=3; torder=5; form=[4,4]; center=[-2.0,0.0,2.0,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neuneu"]; BC=[2,2];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP

## the last entry gives the geometry

## Not dummy variables. Though they can be made some by making appriate changes to the engine_start.jl
#form=[4,4];
#center=[-2.0,0.0,2.0,0.0];
#vinkel=[0,0];
#lenny=[1.0,1.0];
#layer=["dirdir"];
#BC=[1,1]; #should change this.
#OP=1;
#########
#julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP


