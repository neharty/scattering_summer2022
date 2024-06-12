
################# Incidence Wave Block ##################
#Ω=15; 
Ω=9.458170201362927; 
#amp=2.0/sqrt(pi);
amp=1.0
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
#lot=; mu =; 

#g=(xvec,t) -> 
#gp =(xvec,t,norr)-> 
###

#modulate =4.0
#modulate =1.0/0.4
modulate =1.0/0.5193702
g = (xvec,t)->amp*exp(-modulate*(retning(xvec,t))^2);
gp = (xvec,t,nvec)->amp*2*modulate*retning(xvec,t)*exp(-modulate*(retning(xvec,t))^2)*dot(inny,nvec)  ;



################ Geometry Choice Block ####################
## In this case the gemometry and oreientation is simple enough to include in the .sh file. 
#center = 
#blocks =length(center);
#typ=;
#specimen= Int.(typ*ones(blocks))
#vinkel =
#lenny=
#BC =  
#


################ Observation Points Definition Blocks #########

# Example of obs points
n_obs = 4; h_obs=pi/n_obs;
points1=[[6.0, 0.0], [0.0, 6.0], [-6.0, 0.0], [0.0, -6.0]];
#points1=[[-6.0,0.0],[-5.0,0.0],[-5.0,1.5],[-5.0,-1.5],[-1.5,0.0],[1.5,0.0],[5.0,0.0],[5.0,1.5],[5.0,-1.5],[6.0,0.0]];
#points1=[0*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];
#points2=[2.5*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];
#points3=[10*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];
#points4=[11*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];
#points5=[12*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];
#points6=[13*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];

z = vcat(points1);
#z = vcat(points1,points2,points3,points4,points5,points6);


############## Movie plotting options ###################
## number of gridpoints for the x and y axis
Ngridlen = 2^8; 

#left and right endpoints x-axis
x_lenny_left =-4; 
x_lenny_right=4;

#left and right endpoints y-axis
y_lenny_left =-4; 
y_lenny_right=4;

#heatmap ranges
zmin=0.0;
zmax=2.0;
