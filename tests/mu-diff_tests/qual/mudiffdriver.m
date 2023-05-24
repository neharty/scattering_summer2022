
clc;
disp('-----------------------');
disp('    New simulation     ');
disp('-----------------------');
clear all;
close all;

k = 1/(2*pi);
O = readmatrix("cylpos.dat");
a = 1*ones(1,size(O,2));

%loop over N_l values
for i = 0:20
    %M_modes = FourierTruncation(a, k, 'Min', 15);
    M_modes = i*ones(1, 41);
    % Field on grid (based on an integral equation representation)
    %SEE BEGINING FOR XXmin, XXmax, YYmin and YYmax values
    pts = readmatrix("mpos.dat");
    XX = pts(1, :);
    YY = pts(2, :);
    [X,Y] = meshgrid(XX,YY);
    Uinc = PlaneWave(O, a, M_modes, k, pi);
    L = SingleLayer(O, a, M_modes, k);
    rho = L \ Uinc;
    U = ExternalPotential(X, Y, O, a, M_modes, k, rho, [1, 0], 'OnBoundary', 1);
    
    path = join(['~/scattering/tests/mu-diff_tests/qual/data/ka=recip2pi/mydat/', int2str(i)]);
    
    %save the results
    writematrix(real(U), join([path, '/Ur.csv']));
    writematrix(imag(U), join([path, '/Ui.csv']));
end
%writematrix(real(U_tot), '~/scattering/tests/mu-diff_tests/qual/data/U_totr.csv');
%writematrix(imag(U_tot), '~/scattering/tests/mu-diff_tests/qual/data/U_toti.csv');