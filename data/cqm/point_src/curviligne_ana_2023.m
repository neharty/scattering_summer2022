% Didier Dragna - 23/10/2023

clear 
close all

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultLegendInterpreter','latex')

np = input('1. rigid - 2. M120 - 3. M12L02: ');

%% Parameters 

switch np

    case 1
        
        AS = [];
        lambdaS = [];
        
    case 2
    
        AS = [1.4144e+06   1.0014e+06   -3.336e+06   5.2545e+06   3.0317e+07];
        lambdaS = [52.33       494.61       1702.5       1832.7        34000];
    case 3

        AS = [-2.3877e+05   1.3051e+06        91175  -4.4016e+06   2.7118e+07];
        lambdaS = [1.4464       2.1303       1263.8       6531.1        27199];
    
    otherwise 
        
    disp('Not a possible value -> exit')
   
    return

end

NS = length(AS);
Rc = 0.5;
xS = 4;
c0 = 340;
rho0 = 1.21;
epsilon = 100;

BWx = 0.2;
dxi = 0.0098175;
dt = 2.31e-05;

Nfft = 8192;
t = (0:(Nfft-1))*dt;

r_rec = 5;

Nbp = 10;

fs = 1/dt;
Nfft = 8192;
freq = (0:Nfft/2)/Nfft*fs; % frequency
omega = 2*pi*freq; % angular pulsation
k = omega/c0; % wavenumber

%% theta = pi/2.

theta_rec = pi/2;

if (NS==0)
    Beta = zeros(1,length(omega));
else    
    Zimp = zeros(1,length(omega));
    for iloop = 1:NS
        Zimp = Zimp+AS(iloop)./(lambdaS(iloop)-1i*omega);
    end
    Beta = rho0*c0./Zimp;
end

Cpex = zeros(Nbp,length(omega));
for ploop=0:Nbp-1
    Cpex(ploop+1,:) = -2*besselh(ploop,1,k*xS).*(( ploop./(k*Rc)+1i*Beta).*besselj(ploop,k*Rc)-besselj(ploop+1,k*Rc))./...
        (( (ploop./(k*Rc)+1i*Beta).*besselh(ploop,1,k*Rc)-besselh(ploop+1,1,k*Rc)));
end

Cpex(1,:)=Cpex(1,:)/2;

for ploop=1:Nbp
    iNan=find(isnan(Cpex(ploop,:)));
    Cpex(ploop,iNan)=0;
end

% Calcul de la pression analytique

B = BWx/sqrt(log(2));
q_omega = 1i*epsilon*pi*B^2/c0*k.*exp(-k.^2*B^2/4);

pd_dpropa_f = zeros(Nbp,length(omega));
pi_dpropa_f = zeros(1,length(omega));

for ploop = 1:Nbp
    pd_dpropa_f(ploop,2:end) = -1i*Cpex(ploop,2:end).*besselh(ploop-1,1,k(2:end)*r_rec)*cos((ploop-1)*theta_rec)/4;
    iNan2 = find(isnan(pd_dpropa_f(ploop,:)));
    pd_dpropa_f(ploop,iNan2) = 0;
end

p_dpropa_tot = zeros(Nbp+1,length(omega));

pi_dpropa_f(2:end) = -1i*besselh(0,1,k(2:end)*sqrt(r_rec^2+xS^2-2*r_rec*xS*cos(theta_rec)))/4;

p_dpropa_tot(1,:) = q_omega.*pi_dpropa_f;

for iloop=1:length(omega)
    p_dpropa_tot(2:Nbp+1,iloop) = q_omega(iloop).*pd_dpropa_f(1:Nbp,iloop);
end

p_dpropa_fin = zeros(Nbp+1,Nfft);

for ploop = 1:Nbp+1
    p_dpropa_fin(ploop,:)= fft(p_dpropa_tot(ploop,:),Nfft)/Nfft*fs;
end


figure(60)
plot(c0*(0:Nfft-1)*dt/(Rc),2*real(sum(p_dpropa_fin(:,:)))/epsilon,'-r','LineWidth',2)
set(gca,'FontSize',20)
xlim([5.5 12]*2)
ylim([-0.035 0.065])
set(gca,'YTick',-0.03:0.03:0.06)
xlabel('$t$')
ylabel('$p$')
set(gca,'TickLabelInterpreter','latex')
hold on
 
 %% theta = 3pi/4

theta_rec = 3*pi/4;

if (NS==0)
    Beta = zeros(1,length(omega));
else   
    Zimp = zeros(1,length(omega));
    for iloop = 1:NS
        Zimp = Zimp+AS(iloop)./(lambdaS(iloop)-1i*omega);
    end
    Beta = rho0*c0./Zimp;
end

Cpex = zeros(Nbp,length(omega));
for ploop=0:Nbp-1
    Cpex(ploop+1,:) = -2*besselh(ploop,1,k*xS).*(( ploop./(k*Rc)+1i*Beta).*besselj(ploop,k*Rc)-besselj(ploop+1,k*Rc))./...
        (( (ploop./(k*Rc)+1i*Beta).*besselh(ploop,1,k*Rc)-besselh(ploop+1,1,k*Rc)));
end

Cpex(1,:)=Cpex(1,:)/2;

for ploop=1:Nbp
    iNan=find(isnan(Cpex(ploop,:)));
    Cpex(ploop,iNan)=0;
end

% Calcul de la pression analytique

B=BWx/sqrt(log(2));
q_omega=1i*epsilon*pi*B^2/c0*k.*exp(-k.^2*B^2/4);

pd_dpropa_f=zeros(Nbp,length(omega));
pi_dpropa_f=zeros(1,length(omega));

for ploop=1:Nbp
    pd_dpropa_f(ploop,2:end)=-1i*Cpex(ploop,2:end).*besselh(ploop-1,1,k(2:end)*r_rec)*cos((ploop-1)*theta_rec)/4;
    iNan2=find(isnan(pd_dpropa_f(ploop,:)));
    pd_dpropa_f(ploop,iNan2)=0;
end

p_dpropa_tot=zeros(Nbp+1,length(omega));

pi_dpropa_f(2:end)=-1i*besselh(0,1,k(2:end)*sqrt(r_rec^2+xS^2-2*r_rec*xS*cos(theta_rec)))/4;

p_dpropa_tot(1,:)=q_omega.*pi_dpropa_f;
for iloop=1:length(omega)
    p_dpropa_tot(2:Nbp+1,iloop)=q_omega(iloop).*pd_dpropa_f(1:Nbp,iloop);
end

p_dpropa_fin=zeros(Nbp+1,Nfft);
for ploop =1:Nbp+1
    p_dpropa_fin(ploop,:)= fft(p_dpropa_tot(ploop,:),Nfft)/Nfft*fs;
end

 
figure(61)
plot(c0*(0:Nfft-1)*dt/(Rc),2*real(sum(p_dpropa_fin(:,:)))/epsilon,'-r','LineWidth',2)
set(gca,'FontSize',20)
xlim([7.5 11.5]*2)
ylim([-0.035 0.065])
set(gca,'YTick',-0.03:0.03:0.06)
xlabel('$t$')
ylabel('$p$')
set(gca,'TickLabelInterpreter','latex')
hold on
 
%% theta = pi

theta_rec = pi;

if (NS==0)
    Beta = zeros(1,length(omega));
else    
    Zimp = zeros(1,length(omega));
    for iloop = 1:NS
        Zimp = Zimp+AS(iloop)./(lambdaS(iloop)-1i*omega);
    end
    Beta = rho0*c0./Zimp;
end

Cpex = zeros(Nbp,length(omega));
for ploop=0:Nbp-1
    Cpex(ploop+1,:) = -2*besselh(ploop,1,k*xS).*(( ploop./(k*Rc)+1i*Beta).*besselj(ploop,k*Rc)-besselj(ploop+1,k*Rc))./...
        (( (ploop./(k*Rc)+1i*Beta).*besselh(ploop,1,k*Rc)-besselh(ploop+1,1,k*Rc)));
end

Cpex(1,:)=Cpex(1,:)/2;

for ploop=1:Nbp
    iNan=find(isnan(Cpex(ploop,:)));
    Cpex(ploop,iNan)=0;
end

% Calcul de la pression analytique


B = BWx/sqrt(log(2));
q_omega = 1i*epsilon*pi*B^2/c0*k.*exp(-k.^2*B^2/4);

pd_dpropa_f = zeros(Nbp,length(omega));
pi_dpropa_f = zeros(1,length(omega));

for ploop = 1:Nbp
    pd_dpropa_f(ploop,2:end) = -1i*Cpex(ploop,2:end).*besselh(ploop-1,1,k(2:end)*r_rec)*cos((ploop-1)*theta_rec)/4;
    iNan2 = find(isnan(pd_dpropa_f(ploop,:)));
    pd_dpropa_f(ploop,iNan2) = 0;
end

p_dpropa_tot=zeros(Nbp+1,length(omega));

pi_dpropa_f(2:end) = -1i*besselh(0,1,k(2:end)*sqrt(r_rec^2+xS^2-2*r_rec*xS*cos(theta_rec)))/4;

p_dpropa_tot(1,:) = q_omega.*pi_dpropa_f;
for iloop = 1:length(omega)
    p_dpropa_tot(2:Nbp+1,iloop) = q_omega(iloop).*pd_dpropa_f(1:Nbp,iloop);
end

p_dpropa_fin = zeros(Nbp+1,Nfft);
for ploop =1:Nbp+1
    p_dpropa_fin(ploop,:) = fft(p_dpropa_tot(ploop,:),Nfft)/Nfft*fs;
end

figure(62)
plot(c0*(0:Nfft-1)*dt/(Rc),2*real(sum(p_dpropa_fin(:,:)))/epsilon,'-r','LineWidth',2)
set(gca,'FontSize',20)
xlim([8 12]*2)
ylim([-0.02 0.05])
xlabel('$t$')
ylabel('$p$')
set(gca,'TickLabelInterpreter','latex')
hold on
