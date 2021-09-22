function [W, error_values] = NNK_OMP(A, K_dd, K_yd, knn_param, K_yy, reg)
if nargin < 6
    reg=1e-6;
end
G_dd = A'*K_dd*A; % Size #atoms x #atoms
G_dy = A'*K_yd'; % Size #atoms x # data points
mask = find_knn_mask(G_dy', knn_param)';

[m, n] = size(G_dy);
error_values = zeros(n, 1);
W = zeros(m, n);
for i = 1:n
    nodes_i = find(mask(:, i)); % only consider similar enough neighbors
    G_i = full(G_dd(nodes_i, nodes_i)); % + eye(length(nodes_i)); 
    g_i = full(G_dy(nodes_i, i)); % node to dictionary similarity
    qp_output = nonnegative_qp_solver(G_i, g_i, reg, g_i);
    qpsol = qp_output.xopt;
    W(nodes_i, i) = qpsol; %/sum(qpsol);
    error_values(i) = (1 - 2*qpsol'*g_i + qpsol'*G_i*qpsol);
end
end

function mask = find_knn_mask(G, k)
[n, m] = size(G);
if k < m
    mask = G;
    for it = 1:n
        % sort points according to similarities: 
        [~, order] = sort(G(it,:), 'descend'); 

        % for all points which are not among the k nearest neighbors, set mask to 0: 
        mask(it, order(k+1:end)) = 0;
        mask(it, order(1:k)) = 1; %unweighted! 
    end
else
    mask = G > 0;
end
end