

if(OPENING==1)
    σ=12.0; prepre="samesame" ;
elseif(OPENING==2)
    σ=24.0; prepre = "smaller";
elseif(OPENING==3)
    σ=3.0; prepre = "larger";
end

################# Incidence Wave Block ##################
Ω=6; 
amp=1.0;
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
lot=4; mu =1; 
smooth_cut=(x,t)->( (abs(retning(x,t))<lot)*exp(-mu/((retning(x,t)-lot)^2 *(retning(x,t)+lot)^2 ) ) + (abs(retning(x,t))>=lot)*(0) );

##λλ=2.8;#2.2;
#λλ=(pi/2)/1.7#1.4;
#wm = 2*pi*1/λλ;
##why am I not smooth cutting here?  
#g=(xvec,t) -> real(im/(im+(wm/4)*retning(xvec,t))^5)*smooth_cut(xvec,t);
#gp =(xvec,t,norr)-> real((5/4)*wm*im/(im+(wm/4)*retning(xvec,t))^6)*smooth_cut(xvec,t);
###



g = (xvec,t)->amp*exp(-0.5(retning(xvec,t)/(1.0/σ))^2);
gp = (xvec,t,nvec)->amp*(retning(xvec,t)/(1.0/σ))*exp(-0.5*(retning(xvec,t)/(1/σ))^2)*dot(inny,nvec)  ;



################ Geometry Choice Block ####################
# In this case the gemometry and oreientation is simple enough to include in the .sh file. 
center=[[i-0.5,j] for i=-3:4 ,j=-2:2];
blocks =length(center);
specimen = Int.(ones(blocks));
vinkel = [(i-1)*pi/sqrt(2) for i=1:blocks];
lenny = [0.25 for i=1: blocks];
BC = Int.(BBC[1]*ones(blocks));

#center=[[-1,0],[1,0]];
#blocks =length(center);
#specimen = 4*Int.(ones(blocks));
#vinkel = [0.0 for i=1:blocks];
#lenny = [0.5 for i=1: blocks];
#BC = Int.(BBC[1]*ones(blocks));
#




################ Observation Points Definition Blocks #########
points1=[[-4+2*i,2.5] for i=0:4];
points2=[[-4+2*i,-2.5] for i=0:4];

points3=[[-4+2*i,5.0] for i=0:4];
points4=[[-4+2*i,-5.0] for i=0:4];


z = vcat(points1,points2,points3,points4)


############# for movie plotting purposes only #############
##number of grdipoints for each axis. 
Ngridlen = 2^8; ;

## x-axis range
x_lenny_left= -4;
x_lenny_right= 4;

##y-axis range
Y_lenny_left= -4;
Y_lenny_right= 4;

## heatmap output range. 
zmin=0.0;
zmax=1.5;
