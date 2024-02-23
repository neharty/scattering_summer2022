centerstr = load('centers.mat', 'cen');
center = centerstr.cen;
radius = ones(length(center), 1);
Anpstr = load('scmat.mat', 'A');
Anp = Anpstr.A;
k = 0.3471651766977402j;
A = MultipoleManyCirclesReducedModified(k, center, radius, 7);
%Aold = MultipoleManyCirclesReduced(k, center, radius, 7);
%%
max(abs(A), [], 'all')
% max(abs(Anp), [], 'all')
% cond(Anp)
% diff = abs(A - Anp);
% max(diff, [], 'all')
% rhs = rand(100*15);
% x1 = A\rhs;
% x2 = Anp\rhs;
% x0 = Aold\rhs;
% max(abs(x1-x2), [], 'all')
% max(abs(x2-x0), [], 'all')
% imagesc(diff);
% colorbar;
