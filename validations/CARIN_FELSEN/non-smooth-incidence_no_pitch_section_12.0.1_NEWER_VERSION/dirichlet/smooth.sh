xx=8; 
tt=14; 
y=5;
z=1;
T=40;
Inc=11; 
torder=5;
form=[1,1,1,1,1];
center=[0.0,0.0];
vinkel=[1];
lenny=[0.5,0.5,0.5,0.5,0.5];
layer=["dir"];
BC=[1,1,1,1,1];
OP=1;
julia MIX_bdry_cq.jl 1 $torder $T $xx $tt $y $z $Inc $form $center $vinkel $lenny 0 $layer $BC $OP

