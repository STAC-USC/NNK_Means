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