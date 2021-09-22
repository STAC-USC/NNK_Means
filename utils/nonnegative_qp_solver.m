function[out] = nonnegative_qp_solver(A, b, inner_tol, x_init)

% initialization
n = size(A, 1);
AtA = A;%+1e-12*eye(n);
Atb = b;

maxiter = 50*n;
if nargin < 4
    x_init = zeros(n, 1);
end

x = x_init;    %Intial value for solution
iter = 0;

F = true(n, 1);    % F tracks elements pruned based on negativity  

check = 1; %intial value for first run

tol = inner_tol;

while (check > tol) && (iter < maxiter)
    fH1 = x > eps;
    F = F & fH1;

    R = chol(AtA(F,F));

    x = zeros(n, 1);
    x(F) =  R \ (R' \ Atb(F));

    iter = iter + 1;
    
    N = x < eps;
    if sum(N) > 0   % sum of all negative values % positivity constraints check
        check = max(abs(x(N)));
    else
        check = 0;
    end
end

x(x<inner_tol) = 0;
out.xopt = x;
out.check = check;
end