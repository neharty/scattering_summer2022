function[A, B] = alpert_modified(wavenumber,n)

% m=5; ordera = 3;
% 
% chi_orig=[4.004884194926570e-03;
%     7.745655373336686e-02;
%     3.972849993523248e-01;
%     1.075673352915104e+00;
%     2.003796927111872e+00];
% 
% w_orig =[1.671879691147102e-02;
%     1.636958371447360e-01;
%     4.981856569770637e-01;
%     8.372266245578912e+00;
%     9.841730844088381e+00];
% 
% m= 2; ordera =2;
% chi_orig=[1.150395811972836e-01;
%     9.365464527949632e-01];
% w_orig=[3.913373788753340e-01;
%     1.108662621124666e+00];
% Morder = m+3;

% m= 3; ordera =2;
% chi_orig=[2.379647284118974e-02;
%     2.935370741501914e-01;
%     1.023715124251890e+00];
% w_orig=[8.795942675593887e-02;
%     4.989017152913699e-01;
%     9.131388579526912e-01];

% m=4; ordera=3;
% chi_orig=[2.339013027203800e-02;
%     2.854764931311984e-01;
%     1.005403327220700e+00;
%     1.994970303994294e+00];
% 
% w_orig=[8.609736556158105e-02;
%     4.847019685417959e-01;
%     9.152988869123725e-01;
%     1.013901778984250e+00];
% Morder = m+3;

% m=5; ordera=3;
% chi_orig=[4.004884194926570e-03;
%     7.745655373336686e-02;
%     3.972849993523248e-01;
%     1.075673352915104e+00;
%     2.003796927111872e+00];
% 
% w_orig=[1.671879691147102e-02;
%     1.636958371447360e-01;
%     4.981856569770637e-01;
%     8.372266245578912e-01;
%     9.841730844088381e-01];
% 
% m=10; ordera=6;
% chi_orig=[1.175089381227308e-03;
%     1.877034129831289e-02;
%     9.686468391426860e-02;
%     3.004818668002884e-01;
%     6.901331557173356e-01;
%     1.293695738083659e+00;
%     2.090187729798780e+00;
%     3.016719313149212e+00;
%     4.001369747872486e+00;
%     5.000025661793423e+00];
% 
% w_orig=[4.560746882084207e-03;
%     3.810606322384757e-02;
%     1.293864997289512e-01;
%     2.884360381408835e-01;
%     4.958111914344961e-01;
%     7.077154600594529e-01;
%     8.741924365285083e-01;
%     9.661361986515218e-01;
%     9.957887866078700e-01;
%     9.998665787423845e-01];
% Morder = m +3;

m=1; ordera=1;
chi_orig = 1.591549430918953e-01;
w_orig = 0.5;
Morder = m+3;

chi = zeros(2*m,1); w = zeros(2*m,1);
for j=1:m
    chi(j) = chi_orig(j);
    chi(m+j) = - chi_orig(j);
    w(j) = w_orig(j);
    w(m+j) = w_orig(j);
end;

%single layer density
% matrix A_reg
dd = pi/n;
t = 0:dd:2*pi-dd;
%single layer density
% matrix A_reg
J = [0 -1; 1 0]; % rotation matrix
%xp = position_vec(geometry, t);
position = @(tval) [cos(tval); sin(tval)]';
xp = [cos(t); sin(t)]';
%x_t = derivative_vec(geometry, t);
x_t = [-sin(t); cos(t)]';
size(x_t)
%n_t = x_t * J;
arclength = sqrt(x_t(:,1).^2 + x_t(:, 2).^2);

Diff = zeros(2*n, 2*n, 2); 
Dist = zeros(2*n); SinTTau = zeros(2*n);
for i=1:(2*n)
    for j=1:(2*n)
        v= xp(i,:) - xp(j, :);
        Diff(i,j,:) = v;
        Dist(i,j) = norm(v, 2);
    end;
end;

Dist = Dist + diag(ones(2*n, 1));
bess_h_s0 = 1i/2 * dd * besselh(0, wavenumber * Dist);

W = bandmatrix(2*n,ordera);

llindex = @(i, j, n) mod(j-i, n);

% function L = Lagrange(np, list_p, chip)
%     tmp = 1;
%     for i=1:Morder
%         
%     end;
% end;

A = bess_h_s0 .* W; % first part of the matrix
B = zeros(2*n);
for i=1:2*n
    xpi = xp(i,:);
        for p=1:2*m
            chip = chi(p);
            list_p = round(chip-Morder/2):1:round(chip-Morder/2)+Morder-1;
            for j=1:2*n
            containp = any(list_p(:)==llindex(i,j,2*n));
            if containp
                np = llindex(i,j,2*n) - round(chip-Morder/2) +1;
                % L = Lagrange(np,list_p,chip);
                xpi_shift = position(t(i) + chip * dd);
                dist_shift = norm(xpi - xpi_shift, 2);
       
                % B(i,j) = B(i,j) + besselh(0, wavenumber * dist_shift) * w(p) * L;
            end;
        end;
    end;
end;
B = 1i/2 * dd * B;

A = 0.5*A;
B = 0.5*B;
