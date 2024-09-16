
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


amp=1.0;
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
Ω=10.0; 

λ0=6.4
λλ=2.2/λ0;
wm = 2*pi*1/λλ;
g=(xvec,t) -> real(im/(im+(wm/4)*retning(xvec,t))^5)#*(abs(real(im/(im+pi*retning(xvec,t))^5))>=1e-9);
#g=(xvec,t) -> real(im/(im+pi*retning(xvec,t))^5)



################ Geometry Choice Block ####################
## In this case the gemometry and oreientation is simple enough to include in the .sh file. 


width = 1;
center = [[width*(5/3)*i-(5/6)*width,0] for i=-9:10];
blocks =length(center);
typ=1;
specimen= typ*ones(blocks)
pari,arc_l,normv,co = set_geo(specimen);
vinkel =[0.0 for i=1:blocks]
lenny=(width/2)*ones(blocks);




################ Observation Points Definition Blocks #########
z= [[0,(5/3)*width]];

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

