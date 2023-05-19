
disp('-----------------------');
disp('    New simulation     ');
disp('-----------------------');
clear all;
close all;
TypeOfProblem = 'Dirichlet';

% Wavenumber
k = 1/(2*pi);

%% Geometry
O = readmatrix("cylpos.dat");
a = 1*ones(1,size(O,2));

Solution = DirichletSolver(O, a, k, 'PlaneWave', pi);  

% Field on grid (based on an integral equation representation)
%SEE BEGINING FOR XXmin, XXmax, YYmin and YYmax values
pts = readmatrix("mpos.dat");
XX = pts(1, :);
YY = pts(2, :);
[X,Y] = meshgrid(XX,YY);

[U_tot, U] = DirichletNearField(Solution, X, Y);

%save the results
writematrix(real(U), '~/scattering/tests/mu-diff_tests/qual/data/Ur.csv');
writematrix(imag(U), '~/scattering/tests/mu-diff_tests/qual/data/Ui.csv');
writematrix(real(U_tot), '~/scattering/tests/mu-diff_tests/qual/data/U_totr.csv');
writematrix(imag(U_tot), '~/scattering/tests/mu-diff_tests/qual/data/U_toti.csv');