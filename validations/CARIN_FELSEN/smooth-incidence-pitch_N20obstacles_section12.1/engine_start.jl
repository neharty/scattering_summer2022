
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
Ω=10.0; 
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
lot=8; mu =1; 
smooth_cut=(x,t)->( (abs(retning(x,t))<lot)*exp(-mu/((retning(x,t)-lot)^2 *(retning(x,t)+lot)^2 ) ) + (abs(retning(x,t))>=lot)*(0) );

#λ0=6.4
#λλ=1.35/λ0;
λλ = 3/15;
wm = 2*pi*1/λλ;
g=(xvec,t) -> real(im/(im+(wm/4)*retning(xvec,t))^5)*smooth_cut(xvec,t);


################ Geometry Choice Block ####################
## In this case the gemometry and oreientation is simple enough to include in the .sh file. 

dis=1.0;
width=λλ*1.5;
ϕϕ=pi/2;
re_ce = (width/2)*[sin(ϕϕ),cos(ϕϕ)]

center = [[dis*i/2,0]+re_ce for i=0:19]
blocks =length(center);
hold = [[dis*(i),0] for i=0:9];
vinkel =[(-1)^(i-1) *(pi/2-ϕϕ) for i=1:blocks]
typ=1;
specimen= typ*ones(blocks)
pari,arc_l,normv,co  = set_geo(specimen);
lenny=(width/2)*ones(blocks);



################ Observation Points Definition Blocks #########
z= [[0,50/3]]; #this is the original observation point in the paper, which did not fit their results. 
## the other observations points are all of which did not match either. 
        #local z= [[0,(10/3)*width *35.5]];
        #local z= [[0,dis*5/3]];
        #local z= [[0,7]];
        #local z= [[0,1.25]];
        #local z= [[0,17.5]];
    
        #local z= [[4.9,width*5/3]];
        


############### Movie plotting options ###############
##number of gridpoints x and y-axis. 
Ngridlen = 2^8; 

# left and right endpoints (x-axis)
x_lenny_left =0;
x_lenny_right=10;

# left and right endpoints (y-axis)
y_lenny_left=-4;
y_lenny_right=4;

## heatmap ranges
zmin=0.0;
zmax=2.0;

