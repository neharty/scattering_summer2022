N = 41;
dx = 1/N;
dt = 0.1;
k = 1/6;
a = k*dt/dx^2;
A = zeros(N-1,N-1);
A(1,1) = 1+2*a; A(1,2)=-a;
for i=2:N-2
    A(i,i)=1+2*a;
    A(i-1,i)=-a;
    A(i,i-1)=-a;
end
A(N-2,N-1)=-a; A(N-1,N-1)=1+2*a; A(N-1,N-2)=-a;

x = dx:dx:1-dx;
f = @(x) sin(2*pi*x);
b = f(x');
n = length(b);
u = zeros(n,1);

format long
[ugs,numIter] = GS(A, b, u, 1e-3, 1000)

y1 = sin(2*pi*x);
y2 = exp(-4*pi*pi*k*dt)*sin(2*pi*x);

figure(3)
hold on  
plot(x,y1, 'o')
plot(x,y2, '+')
plot(x,ugs, 'x') 

u = zeros(n,1);

format long
[uj,numIter] = Jacobi(A, b, u, 1e-3, 1000)

plot(x,uj, '--')

xlabel('u');
ylabel('x');
legend('t=0','t=0.1 (exact)','t=0.1 (computed, GS)','t=0.1 (computed, Jacobi)')
hold off

u = zeros(n,1);
[u,numIter] = SOR(A, b, u, 1e-3, 1000, 1.875)

u = zeros(n,1);
[u,numIter] = conjGradient(A,b,u,1.0e-15)

function [x,numIter] = GS(A, b, x, epsilon, maxIter)
% Guass-Seidel method for A*x = b.
% USAGE: [x,numIter] = GS(A,b,x,epsilon,maxIter)
% INPUT:
% A=        n x n matrix
% b=        n-dimensional vector
% epsilon=  error tolerance (1.0e-9 is the default) 
% maxIter=  allowable number of iterations (500 is the default)
% x=        starting solution vector
% OUTPUT:
% x=        approximate solution to A*x = b.

if nargin < 4;  epsilon=1.0e-9; end
if nargin < 5;  maxIter=500; end

rowDom = true;
for r = 1:size(A,1)
    rowDom = 2 * abs(A(r,r)) > sum(abs(A(r,:)));
end

if rowDom == 0; disp(['Matrix A is not diagonally-dominant']); end

n=size(x,1);
normVal=Inf; 

numIter=0;
while normVal>epsilon
    xold=x;
    for i=1:n
        sigma=0;     
        for j=1:i-1
            sigma=sigma+A(i,j)*x(j);
        end
        for j=i+1:n
            sigma=sigma+A(i,j)*xold(j);
        end        
        x(i)=(b(i)-sigma)/A(i,i);
     end
    numIter=numIter+1;
    normVal=norm((xold-x),Inf);
    
    figure(1)
    hold on
    plot(numIter,normVal, 'x')
    title('Gauss-Seidel Method');
    xlabel('# of iterations');
    ylabel('error');
    
    if numIter>maxIter error('Too many iteration'); return; end
end
hold off
    
%%
fprintf('Gauss Seidel Error Compared to MATLAB Built-in Function\n');
e1 = norm((x-linsolve(A,b)),1)
e2 = norm((x-A\b),1)
e3 = norm((x-pinv(A)*b),1)

end

function [x,numIter] = Jacobi(A, b, x, epsilon, maxIter)
% Jacobi method for A*x = b.
% USAGE: [x,numIter] = Jacobi(A,b,x,epsilon,maxIter)
% INPUT:
% A=        n x n matrix
% b=        n-dimensional vector
% epsilon=  error tolerance (1.0e-9 is the default) 
% maxIter=  allowable number of iterations (500 is the default)
% x=        starting solution vector
% OUTPUT:
% x=        approximate solution to A*x = b.

if nargin < 4;  epsilon=1.0e-9; end
if nargin < 5;  maxIter=500; end

n=size(x,1);
normVal=Inf; 

numIter=0;
while normVal>epsilon
    xold=x;
    for i=1:n
        sum=0;     
        for j=1:n
            if j~=i
               sum=sum+A(i,j)*xold(j);
            end           
        end
        x(i)=(b(i)-sum)/A(i,i);
    end
    numIter=numIter+1;
    normVal=norm((xold-x),Inf);
    if numIter>maxIter error('Too many iteration'); return; end
end
%%
fprintf('Jacobi Method Errors Compared to MATLAB Built-in Function\n');
e1 = norm((x-linsolve(A,b)),1)
e2 = norm((x-A\b),1)
e3 = norm((x-pinv(A)*b),1)

end

function [x,numIter] = SOR(A, b, x, epsilon, maxIter, omega)
% Guass-Seidel method for A*x = b.
% USAGE: [x,numIter] = GS(A,b,x,epsilon,maxIter)
% INPUT:
% A=        n x n matrix
% b=        n-dimensional vector
% epsilon=  error tolerance (1.0e-9 is the default) 
% maxIter=  allowable number of iterations (500 is the default)
% omega  =  successive over relaxation coefficient
% x=        starting solution vector
% OUTPUT:
% x=        approximate solution to A*x = b.

if nargin < 4;  epsilon=1.0e-9; end
if nargin < 5;  maxIter=500; end

rowDom = true;
for r = 1:size(A,1)
    rowDom = 2 * abs(A(r,r)) > sum(abs(A(r,:)));
end

if rowDom == 0; disp(['Matrix A is not diagonally-dominant']); end

n=size(x,1);
normVal=Inf; 

numIter=0;
while normVal>epsilon
    xold=x;
    for i=1:n
        sigma=0;     
        for j=1:i-1
            sigma=sigma+A(i,j)*x(j);
        end
        for j=i+1:n
            sigma=sigma+A(i,j)*xold(j);
        end        
        x(i)=(b(i)-sigma)/A(i,i);
        x(i)=omega*x(i)+(1-omega)*xold(i);
     end
    numIter=numIter+1;
    normVal=norm((xold-x),Inf);
    
    figure(2)
    hold on
    plot(numIter,normVal, 'x')
    title('SOR Method');
    xlabel('# of iterations');
    ylabel('error');
    
    if numIter>maxIter error('Too many iteration'); return; end
end
hold off
    
%%
fprintf('SOR Error Compared to MATLAB Built-in Function\n');
e1 = norm((x-linsolve(A,b)),1)
e2 = norm((x-A\b),1)
e3 = norm((x-pinv(A)*b),1)

end

function [x,numIter] = conjGradient(A,b,x,epsilon)
% Solves Ax=b by conjugate gradient method.
% USAGE: [x,numIter] = conjGrad(A,b,x,epsilon)
% INPUT:
% A        =  n × n matrix A which is symmetric and positive-definite
% b        =  constant right hand side vector
% x        =  starting solution vector
% epsilon  =  error tolerance (default = 1.0e-9)
% OUTPUT:
% x        =  solution vector
% numIter  =  number of iterations

%if nargin < 4;  epsilon=1.0e-9; end
n = length(b);
r = b - A*x;
p = r;

for numIter = 1:n
    Ap = A*p;
    alpha = dot(p,r)/dot(p,Ap);
    x = x + alpha*p;
    r = b - A*x;
    if sqrt(dot(r,r)) < epsilon
        return
    else
        beta = -dot(r,Ap)/dot(p,Ap);
        p = r + beta*p;
    end
end
%error('Too many iterations')
%%
fprintf('Conjugate Gradient Method Error Compared to MATLAB Built-in Function\n');
e1 = norm((x-linsolve(A,b)),1)
e2 = norm((x-A\b),1)
e3 = norm((x-pinv(A)*b),1)

end
