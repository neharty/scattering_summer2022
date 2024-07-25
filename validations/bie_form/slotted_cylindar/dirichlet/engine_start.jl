
################# Incidence Wave Block ##################
Ω=6; 
amp=1.0;
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
lot=4; mu =1; 
smooth_cut=(x,t)->( (abs(retning(x,t))<lot)*exp(-mu/((retning(x,t)-lot)^2 *(retning(x,t)+lot)^2 ) ) + (abs(retning(x,t))>=lot)*(0) );

#λλ=2.8;#2.2;
λλ=(pi/2)/1.7#1.4;
wm = 2*pi*1/λλ;
#why am I not smooth cutting here?  
g=(xvec,t) -> real(im/(im+(wm/4)*retning(xvec,t))^5)*smooth_cut(xvec,t);
gp =(xvec,t,norr)-> real((5/4)*wm*im/(im+(wm/4)*retning(xvec,t))^6)*smooth_cut(xvec,t);
##

#modulate =3.0
#g = (xvec,t)->amp*exp(-modulate*(retning(xvec,t))^2);
#gp = (xvec,t,nvec)->amp*modulate*retning(xvec,t)*exp(-modulate*(retning(xvec,t))^2)*dot(inny,nvec)  ;



################ Geometry Choice Block ####################
## In this case the gemometry and oreientation is simple enough to include in the .sh file. 





################ Observation Points Definition Blocks #########
b3 = 2.5;
b4 = 2.5; 
fun = (x)->[b3*cos(x),b4*sin(x)];
N_Obs = 2^5; h_Obs=2*pi/N_Obs;
x=[fun(i*h_Obs)+[0,0] for i=0:N_Obs-1];
y= [ [0.625*cos(i*h_Obs),0.625*sin(i*h_Obs)] for i =0:N_Obs-1];
z = vcat(x,y)


############## Movie plotting options ###################
## number of gridpoints for the x and y axis
Ngridlen = 2^7; 

#left and right endpoints x-axis
x_lenny_left =-4; 
x_lenny_right=4;

#left and right endpoints y-axis
y_lenny_left =-4; 
y_lenny_right=4;

#heatmap ranges
zmin=0.0;
zmax=2.0;
