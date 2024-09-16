
################# Incidence Wave Block ##################
if(ret == 1)
    inny = [1,-1]; dir = "REG";
elseif(ret==2)
    inny = [0,-1]; dir = "UPDOWN";
elseif(ret==3)
    inny = [1,0];  dir = "LEFTRIGHT";
elseif(ret==4)
    θ= -pi/4;
    inny=[cos(θ),sin(θ)]; dir="REGA"
elseif(ret==5)
    θ= -pi/2; dir="ANGLE"
    inny=[cos(θ),sin(θ)];
elseif(ret==6)
    θ= 0; dir="LEFTRIGHTA"
    inny=[cos(θ),sin(θ)];
elseif(ret==11)
    θ= pi+ 7*pi/18; dir="CARIN"
    inny=[cos(θ),sin(θ)];
end


amp =1.0; 
Ω=5.0;
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω; 
aa = 2*pi/0.3;
g1 =(xvec,t)-> -sin(aa*retning(xvec,t))*(0.0<=retning(xvec,t)<0.15);
g2 =(xvec,t)-> -2sin(aa*retning(xvec,t))*(0.15<=retning(xvec,t)<0.3);
g3 =(xvec,t)-> -sin(aa*retning(xvec,t))*(0.3<=retning(xvec,t)<0.45);
g = (xvec,t)-> g1(xvec,t) + g2(xvec,t) + g3(xvec,t);

##removed negative infront of each of the below. 
gp1 =(xvec,t,norr)-> aa*cos(aa*retning(xvec,t))*dot(inny,norr)*(0.0<=retning(xvec,t)<0.15);
gp2 =(xvec,t,norr)-> 2*aa*cos(aa*retning(xvec,t))*dot(inny,norr)*(0.15<=retning(xvec,t)<0.3);
gp3 =(xvec,t,norr)-> aa*cos(aa*retning(xvec,t))*dot(inny,norr)*(0.3<=retning(xvec,t)<0.45);
gp = (xvec,t,norr)-> gp1(xvec,t,norr) + gp2(xvec,t,norr) + gp3(xvec,t,norr);



################ Geometry Choice Block ####################
## In this case the gemometry and oreientation is simple enough to include in the .sh file. 

center = [[(5/3)*i,0] for i=-7:7];
blocks =length(center);
typ=1;
specimen= Int.(typ*ones(blocks))
#pari,arc_l,normv,co = set_geo(specimen,);
vinkel =[0.0 for i=1:blocks]
lenny=0.5*ones(blocks);
BC = Int.(BBC[1]*ones(blocks));




################ Observation Points Definition Blocks #########
z= [[0,50/3]];


############### Movie plotting options ###############
##number of gridpoints x and y-axis. 
Ngridlen = 2^8; 

# left and right endpoints (x-axis)
x_lenny_left =-10;
x_lenny_right=10;

# left and right endpoints (y-axis)
y_lenny_left=-4;
y_lenny_right=4;

## heatmap ranges
zmin=0.0;
zmax=2.0;

