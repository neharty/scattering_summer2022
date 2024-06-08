#!/bin/bash 
#
film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
#
lrep=1;xx=7;tt=14;y=5;z=1;T=40;Inc=3;torder=5;form=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]; center=[-4.5,-4.5,-1.5,-4.5,1.5,-4.5,4.5,-4.5,-4.5,-1.5,-1.5,-1.5,1.5,-1.5,4.5,-1.5,-4.5,1.5,-1.5,1.5,1.5,1.5,4.5,1.5,-4.5,4.5,-1.5,4.5,1.5,4.5,4.5,4.5];vinkel=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; lenny=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]; layer=["neudir"]; BC=[2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neudir_1.txt neudir_lrep1_xx7_Inc3_tt14.txt
lrep=1;xx=7;tt=14;y=5;z=1;T=40;Inc=4;torder=5;form=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]; center=[-4.5,-4.5,-1.5,-4.5,1.5,-4.5,4.5,-4.5,-4.5,-1.5,-1.5,-1.5,1.5,-1.5,4.5,-1.5,-4.5,1.5,-1.5,1.5,1.5,1.5,4.5,1.5,-4.5,4.5,-1.5,4.5,1.5,4.5,4.5,4.5];vinkel=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; lenny=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]; layer=["neudir"]; BC=[2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neudir_1.txt neudir_lrep1_xx7_Inc4_tt14.txt
lrep=2;xx=7;tt=14;y=5;z=1;T=40;Inc=3;torder=5;form=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]; center=[-4.5,-4.5,-1.5,-4.5,1.5,-4.5,4.5,-4.5,-4.5,-1.5,-1.5,-1.5,1.5,-1.5,4.5,-1.5,-4.5,1.5,-1.5,1.5,1.5,1.5,4.5,1.5,-4.5,4.5,-1.5,4.5,1.5,4.5,4.5,4.5];vinkel=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; lenny=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]; layer=["neudir"]; BC=[2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neudir_2.txt neudir_lrep2_xx7_Inc3_tt14.txt
lrep=2;xx=7;tt=14;y=5;z=1;T=40;Inc=4;torder=5;form=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]; center=[-4.5,-4.5,-1.5,-4.5,1.5,-4.5,4.5,-4.5,-4.5,-1.5,-1.5,-1.5,1.5,-1.5,4.5,-1.5,-4.5,1.5,-1.5,1.5,1.5,1.5,4.5,1.5,-4.5,4.5,-1.5,4.5,1.5,4.5,4.5,4.5];vinkel=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; lenny=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]; layer=["neudir"]; BC=[2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neudir_2.txt neudir_lrep2_xx7_Inc4_tt14.txt
