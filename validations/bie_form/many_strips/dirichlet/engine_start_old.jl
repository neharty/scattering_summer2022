
if(OPENING==1)
    σ=12.0; prepre="samesame" ;
elseif(OPENING==2)
    σ=24.0; prepre = "smaller";
elseif(OPENING==3)
    σ=3.0; prepre = "larger";
end




################# Incidence Wave Block ##################
#Ω=6; 
#amp=1.0;
#retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
#lot=4; mu =1; 
#smooth_cut=(x,t)->( (abs(retning(x,t))<lot)*exp(-mu/((retning(x,t)-lot)^2 *(retning(x,t)+lot)^2 ) ) + (abs(retning(x,t))>=lot)*(0) );
#
###λλ=2.8;#2.2;
##λλ=(pi/2)/1.7#1.4;
##wm = 2*pi*1/λλ;
###why am I not smooth cutting here?  
##g=(xvec,t) -> real(im/(im+(wm/4)*retning(xvec,t))^5)*smooth_cut(xvec,t);
##gp =(xvec,t,norr)-> real((5/4)*wm*im/(im+(wm/4)*retning(xvec,t))^6)*smooth_cut(xvec,t);
####

#g = (xvec,t)->amp*exp(-0.5(retning(xvec,t)/(1.0/σ))^2);
#gp = (xvec,t,nvec)->amp*(retning(xvec,t)/(1.0/σ))*exp(-0.5*(retning(xvec,t)/(1/σ))^2)*dot(inny,nvec)  ;

Ω=5.8;
amp=1.0;
modulate=3.0;
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;

g = (xvec,t)->amp*exp(-modulate*(retning(xvec,t))^2)*cos(retning(xvec,t)+Ω);
gp = (xvec,t,nvec)->amp*exp(-modulate*(retning(xvec,t))^2)*(2*modulate*cos(retning(xvec,t)+Ω)*retning(xvec,t) + sin(retning(xvec,t)+Ω) )*dot(inny,nvec)  ;


################ Geometry Choice Block ####################
## In this case the gemometry and oreientation is simple enough to include in the .sh file. 
#center=[[i-0.5,j] for i=-3:4 ,j=-2:2];
#blocks =length(center);
#specimen = Int.(ones(blocks));
#vinkel = [(i-1)*pi/sqrt(2) for i=1:blocks];
#lenny = [0.25 for i=1: blocks];
#BC = Int.(BBC[1]*ones(blocks));
#

#center=[[0,0]];
#blocks =length(center);
#specimen = 1*Int.(ones(blocks));
#vinkel = [0.0 for i=1:blocks];
#lenny = [1.0 for i=1: blocks];
#BC = Int.(BBC[1]*ones(blocks));


center0 = [[i,0.1] for i=-4:2:4]
center1 = [[i,1.1] for i=-3:2:4]
center2 = [[i,-1.1] for i=-3:2:4]
center = vcat(center0,center1,center2);
typ=1;
blocks =length(center);
specimen= Int.(typ*ones(blocks));
vinkel0 =[i for i=1:length(center0)]
vinkel1 =[pi/4 for i=1:length(center1)]
vinkel2 =[-pi/4 for i=1:length(center2)]
vinkel = vcat(vinkel0,vinkel1,vinkel2)
lenny =[0.5 for i=1:blocks]
BC = Int.(BBC[1]*ones(blocks));


################ Observation Points Definition Blocks #########
points1=[[-4+2*i,2.5] for i=0:4];
points2=[[-4+2*i,-2.5] for i=0:4];

points3=[[-4+2*i,5.0] for i=0:4];
points4=[[-4+2*i,-5.0] for i=0:4];

z = vcat(points1,points2,points3,points4);


################ for movie purposes solely ###############
## number of gridpoints for each axis. 
Ngridlen = 2^8; 

#specifies the x and y length for the grid in the movie. 
x_len_left =-5;
x_len_right =5;

y_len_left =-5;
y_len_right =5;

#range output for the heatmap
zmin=0; zmax=1.5; ##the max and min value for the output range. 


