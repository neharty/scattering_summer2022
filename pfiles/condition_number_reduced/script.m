T  = T_final;     % final time
dt = T/N;   % time step
t=(0:dt:T); % time grid
lambda = max([dt^(3/N) eps^(1/2/N)]);  % radius of the circle in the complex plane
zN = exp(2i*pi/(N+1));   % discretization point on the circle
gm = @(z) 0.5*(z.^2-4*z+3); % Z-transform corresponding to BDF2
sl  = gm(zN.^(0:-1:-N)*lambda)/dt;  
kl = 1i*sl; % Complex frequencies
