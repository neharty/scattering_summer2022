center = zeros(400,2);
>> for i=1:20
for j=1:20
center(20 * (i-1) +j, 1) = 3 * 0.05 * (j-1); center(20 * (i-1) +j, 2) = -3 * 0.05 * (i-1);
end;
end;
radius = 0.05 * ones(400,1);


%gmres line
[coeff,flag,relres,iter,res]=gmres(A,-rhs,N_Fourier * N_circles,1e-8);
size(res)