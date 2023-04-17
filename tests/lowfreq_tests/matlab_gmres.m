A = load('A.mat');
A = A.data;
S = svd(A);
b = load('rhs.mat');
b = b.data;
scipy_res = load('res.mat');
scipy_res = scipy_res.data;

[~, flag, ~, iter, matlab_res] = gmres(A, b', 50, 1e-8, 100);
flag
iter
format long
matlab_res
semilogy(scipy_res(1:75), '.-');
hold on;
semilogy(matlab_res, '.-');
