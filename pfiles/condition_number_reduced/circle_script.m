% center = zeros(400,2);
% for i=1:20
% for j=1:20
% center(20 * (i-1) +j, 1) = 3 * 0.05 * (j-1); center(20 * (i-1) +j, 2) = -3 * 0.05 * (i-1);
% end;
% end;
% radius = 0.05 * ones(400,1);
center = zeros(100, 2);
for i=1:10
    for j=1:10
        center(10*(i-1)+j, 1) = 3*(j-1);
        center(10*(i-1)+j, 2) = -3*(i-1);
    end;
end;
radius = ones(100, 1);
k=1.47794707735355580*1e-14+1j*0.4505456673639108844;
%k =  0.45;
[nf, A] = MultipoleManyCirclesReduced(k, center, radius, 7);
%gmres line
%[coeff,flag,relres,iter,res]=gmres(A,-rhs,N_Fourier * N_circles,1e-8);
%size(res)