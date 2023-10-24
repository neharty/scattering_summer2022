%% Normalized impedance spectra for JASA 2013
% Didier Dragna - 23/10/2023

close all
clear

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultLegendInterpreter','latex')

% Physical parameter
rho0 = 1.21; % air density
c0 = 340; % sound speed

freq2 = logspace(log10(10),log10(2000),25);
freq1 = logspace(log10(10),log10(2000),100);
omega = 2*pi*freq1;

% Grass (M120)

ASh = [1.4144e+06   1.0014e+06   -3.336e+06   5.2545e+06   3.0317e+07];
lambdaSh = [52.33       494.61       1702.5       1832.7        34000];
        
Zh = zeros(1,length(freq2));
for ii = 1:length(ASh)
    Zh(:) = Zh(:)+ASh(ii)./(lambdaSh(ii)-1i*2*pi*freq2(:));
end
Zh = Zh/(rho0*c0);

% Exact model
sigmae = 100000; % air flow resistivity
Zh1 = 1 + 0.459*(sigmae./(-1i*rho0*omega)).^0.632;

% Snow (M12L02)

ASs = [-2.3877e+05   1.3051e+06        91175  -4.4016e+06   2.7118e+07];
lambdaSs = [1.4464       2.1303       1263.8       6531.1        27199];

Zs = zeros(1,length(freq2));
for ii = 1:length(ASs)
    Zs(:) = Zs(:)+ASs(ii)./(lambdaSs(ii)-1i*2*pi*freq2(:));
end
Zs = Zs/(rho0*c0);

sigmae = 10000; % air flow resistivity
dlayer = 0.1; % layer thickness
X = freq1/sigmae; 
Zst = 1 + 0.459*(sigmae./(-1i*rho0*omega)).^0.632;
ks = omega/c0.*( 1 + 0.673*(sigmae./(-1i*rho0*omega)).^0.618);
Zs1 = Zst.*coth(-1i*ks*dlayer);


% with dimensions

figure(1)
subplot(211)
semilogx(freq1,real(Zh1),'-k','LineWidth',2)
hold on
semilogx(freq1,real(Zs1),'-r','LineWidth',2)
semilogx(freq2,real(Zh),'.k','MarkerSize',20)
semilogx(freq2,real(Zs),'.r','MarkerSize',20)
plot([50 50],[0.25 30],'--k','LineWidth',2)
plot([1200 1200],[0.25 30],'--k','LineWidth',2)
set(gca,'FontSize',20)
box on
set(gca,'XGrid','on')
set(gca,'YTick',[0 4 8 12])
xlabel('$f$, Hz')
ylabel('${\rm Re}[Z]/\rho_0c_0$')
h=legend('M120','M12L02');
%set(h,'PlotBoxAspectRatio',[1 0.85 2])
set(h,'FontSize',17)
xlim([40 2000])
ylim([0 12])
set(gca,'TickLabelInterpreter','latex')

subplot(212)
semilogx(freq1,imag(Zh1),'-k','LineWidth',2)
hold on
semilogx(freq2,imag(Zh),'.k','MarkerSize',20)
semilogx(freq1,imag(Zs1),'-r','LineWidth',2)
semilogx(freq2,imag(Zs),'.r','MarkerSize',20)
plot([50 50],[0.1 100],'--k','LineWidth',2)
plot([1200 1200],[0.1 100],'--k','LineWidth',2)
box on
set(gca,'FontSize',20)
set(gca,'XGrid','on')
set(gca,'YTick',[0 5 15])
xlabel('$f$, Hz')
ylabel('${\rm Im}[Z]/\rho_0c_0$')
xlim([40 2000])
ylim([0 15])
set(gca,'TickLabelInterpreter','latex')
 h=findobj;
 get(h(2),'Position')
 get(h(1),'ScreenSize')
 set(h(2),'Units','normalized','Position',[0.3 0.1 0.4 0.6])

 % without dimensions

R = 0.5;

figure(10)
subplot(211)
semilogx(freq1*2*pi/c0*R,real(Zh1),'-k','LineWidth',2)
hold on
semilogx(freq1*2*pi/c0*R,real(Zs1),'-r','LineWidth',2)
semilogx(freq2*2*pi/c0*R,real(Zh),'.k','MarkerSize',20)
semilogx(freq2*2*pi/c0*R,real(Zs),'.r','MarkerSize',20)
plot([50 50]*2*pi/c0*R,[0.25 30],'--k','LineWidth',2)
plot([1200 1200]*2*pi/c0*R,[0.25 30],'--k','LineWidth',2)
set(gca,'FontSize',20)
box on
set(gca,'XGrid','on')
set(gca,'YTick',[0 4 8 12])
xlabel('$2\pi f R/c_0$')
ylabel('${\rm Re}[Z]/\rho_0c_0$')
h=legend('M120','M12L02');
%set(h,'PlotBoxAspectRatio',[1 0.85 2])
set(h,'FontSize',17)
xlim([40 2000]*2*pi/c0*R)
ylim([0 12])
set(gca,'TickLabelInterpreter','latex')

subplot(212)
semilogx(freq1*2*pi/c0*R,imag(Zh1),'-k','LineWidth',2)
hold on
semilogx(freq2*2*pi/c0*R,imag(Zh),'.k','MarkerSize',20)
semilogx(freq1*2*pi/c0*R,imag(Zs1),'-r','LineWidth',2)
semilogx(freq2*2*pi/c0*R,imag(Zs),'.r','MarkerSize',20)
plot([50 50]*2*pi/c0*R,[0.1 100],'--k','LineWidth',2)
plot([1200 1200]*2*pi/c0*R,[0.1 100],'--k','LineWidth',2)
box on
set(gca,'FontSize',20)
set(gca,'XGrid','on')
set(gca,'YTick',[0 5 15])
xlabel('$2\pi f R/c_0$')
ylabel('${\rm Im}[Z]/\rho_0c_0$')
xlim([40 2000]*2*pi/c0*R)
ylim([0 15])
set(gca,'TickLabelInterpreter','latex')
 h=findobj;
 get(h(2),'Position')
 get(h(1),'ScreenSize')
 set(h(2),'Units','normalized','Position',[0.3 0.1 0.4 0.6])
