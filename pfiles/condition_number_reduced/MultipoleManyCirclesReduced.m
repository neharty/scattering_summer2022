function[near_field] = MultipoleManyCirclesReduced(wavenumber, center, radius, ntrunc1);
phi = 0; %incident direction
N_circles = length(center);
direction = [cos(phi) sin(phi)];
% Fourier indexing over here
index1 = zeros(2*ntrunc1 + 1,1);
for i=1:ntrunc1+1
    index1(i) = i-1;
end;
for i=ntrunc1+2:2*ntrunc1+1
    index1(i) = i - (2*ntrunc1+2);
end;

%set up right hand side
N_Fourier = 2 * ntrunc1 + 1;
rhs = zeros(N_circles * N_Fourier,1);
for i=1:N_circles
factor = exp(1i * wavenumber * dot(center(i,:), direction));
rhs1 = factor * (1i.^index1) .* exp(-1i * index1 * phi) .* besselj(index1, wavenumber * radius(i)); 
rhs((i-1) * N_Fourier +1 : i * N_Fourier) = rhs1;
end;


%build the matrix
A = eye(N_circles * N_Fourier);


%off diagonal entries
for ell =1:N_circles
    for kk = 1:N_circles
        if (ell ~= kk)
           bcenter = center(ell,:) - center(kk, :); % vector of centers
           distance_center = norm(bcenter,2); % distane between centers
           theta1 = atan2(bcenter(2), bcenter(1)); %the polar angle here
           for i=1:N_Fourier
                for j=1:N_Fourier
                    A((ell-1)*N_Fourier + i, (kk-1)* N_Fourier + j) = besselj(index1(i), wavenumber * radius(ell)) * ...
                        besselh(index1(j) - index1(i), wavenumber * distance_center)/besselh(index1(j), wavenumber * radius(kk)) * ...
                        exp(1i * (index1(j) - index1(i)) * theta1);
                end;
           end;
        end;
    end;
end;


cond(A)
coeff = A\(-rhs);


%compute the near field now
n_near = 256;
x_near = zeros(n_near,2);
for j=1:n_near
    x_near(j,1) = (j-1) *0.2; x_near(j,2) = 3;
end;

near_field = zeros(n_near,1);
for i=1:n_near
    for ell=1:N_circles
    rx1 = x_near(i,:) - center(ell,:); dist1 = norm(rx1,2); th1 = atan2(rx1(2),rx1(1));
    for j=1:N_Fourier
        near_field(i) = near_field(i) + coeff((ell-1) * N_Fourier+ j) * ...
            besselh(index1(j), wavenumber * dist1)/besselh(index1(j), wavenumber * radius(ell)) *...
            exp(1i *index1(j) *th1);
    end;
    end;
end;