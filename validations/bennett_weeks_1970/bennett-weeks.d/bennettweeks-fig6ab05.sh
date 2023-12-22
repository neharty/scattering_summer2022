#!/bin/bash 


film=0 #0 to just evaluate field at set number of points, or 1 to make a movie. 
lrep=1; xx=6 ; tt=14  ; y=5 ;z=1; T=40; Inc=3; torder=5; form=[4,4]; center=[-1.25,0.0,1.25,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neuneu"]; BC=[2,2];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neuneu_1.txt neuneu_1_xx6_tt14-fig6ab05.txt
lrep=1; xx=7 ; tt=14  ; y=5 ;z=1; T=40; Inc=3; torder=5; form=[4,4]; center=[-1.25,0.0,1.25,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neuneu"]; BC=[2,2];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neuneu_1.txt neuneu_1_xx7_tt14-fig6ab05.txt
lrep=1; xx=8 ; tt=14  ; y=5 ;z=1; T=40; Inc=3; torder=5; form=[4,4]; center=[-1.25,0.0,1.25,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neuneu"]; BC=[2,2];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neuneu_1.txt neuneu_1_xx8_tt14-fig6ab05.txt
lrep=1; xx=9 ; tt=14  ; y=5 ;z=1; T=40; Inc=3; torder=5; form=[4,4]; center=[-1.25,0.0,1.25,0.0]; vinkel=[0]; lenny=[1.0,1.0]; layer=["neuneu"]; BC=[2,2];  OP=1; julia MIX_bdry_cq.jl $lrep $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny $film $layer $BC $OP
mv neuneu_1.txt neuneu_1_xx9_tt14-fig6ab05.txt
