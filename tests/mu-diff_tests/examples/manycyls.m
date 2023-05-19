
disp('-----------------------');
disp('    New simulation     ');
disp('-----------------------');
clear all;
close all;
ind_fig = 1;

%% Chose the boundary condition
TypeOfProblem = 'Dirichlet';
%% Wavenumber
k = 2*pi;
%% Geometry
b = 3;
N = 10;     
O = RectangularLattice(b, b, N, N, 'Center', [0, 0]);
a = 1*ones(1,size(O,2));

%% Drawing the circular obstacles
ind_fig = ind_fig +1;
PlotCircles(O, a, ind_fig, 'Color', 'k', 'LineWidth', 2);
axis equal;
xlabel('x'); ylabel('y');
title('Obstacles');

Solution = DirichletSolver(O, a, k, 'PlaneWave', pi);  


%% Field on grid (based on an integral equation representation)
%SEE BEGINING FOR XXmin, XXmax, YYmin and YYmax values
L = 35;
XX = -L:0.1:L;
YY = -L:0.1:L;
[X,Y] = meshgrid(XX,YY);

[U_tot, U] = DirichletNearField(Solution, X, Y);

%% Display on mesh
% Note: to erase white artefacts, use
% set(gcf,'Renderer','Zbuffer');

ind_fig = ind_fig +1; figure(ind_fig);
hold on
surf(X,Y, abs(U_tot));
shading interp;
title(['Absolute value of the total field']);
xlabel('x'); ylabel('y');
view(2); colorbar;
PlotCircles(O, a, ind_fig, 'Color', 'k', 'LineWidth', 2, 'zdata', max(max(abs(U_tot))));
set(gcf,'Renderer','Zbuffer');
hold off

%save the results
writematrix(real(U), '~/scattering/tests/antoine_tests/mu-diff_tests/examples/Ur.csv');
writematrix(imag(U), '~/scattering/tests/antoine_tests/mu-diff_tests/examples/Ui.csv');
writematrix(real(U_tot), '~/scattering/tests/antoine_tests/mu-diff_tests/examples/U_totr.csv');
writematrix(imag(U_tot), '~/scattering/tests/antoine_tests/mu-diff_tests/examples/U_toti.csv');