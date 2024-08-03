
################# Incidence Wave Block ##################
#Ω=7.381808501135772; 
Ω=5.881808501135772;
amp=1;
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
#lot=; mu =; 

#g=(xvec,t) -> 
#gp =(xvec,t,norr)-> 
###

modulate=2.772588722239781;
g = (xvec,t)->amp*exp(-modulate*(retning(xvec,t))^2);
gp = (xvec,t,nvec)->-amp*2*modulate*retning(xvec,t)*exp(-modulate*(retning(xvec,t))^2)*dot(inny,nvec)  ;



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
#n_obs = 10; h_obs=pi/n_obs;
#points1=[[-6.0,0.0],[-3.0,3.0],[-3.0,0.0],[-3.0,-3.0],[0.0,6.0],[0.0,0.0],[0.0,-6.0],[3.0,3.0],[3.0,0.0],[3.0,-3.0],[6.0,0.0]];
#points2=[9*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];
#points3=[10.5*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];
#points4=[12*[cos(i*h_obs),sin(i*h_obs)] for i=0:n_obs];

#z = vcat(points1,points2,points3,points4);

z = [[6.0,0.0],[0.0,6.0],[-6.0,0.0],[0.0,-6.0]];


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
