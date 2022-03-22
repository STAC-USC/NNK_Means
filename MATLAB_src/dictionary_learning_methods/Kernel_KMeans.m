function [A, X] = Kernel_KMeans(params)
% Input : 
%   	  data = column data; dictsize = number of atoms
%   	  Tdata = sparsity level; iternum = number of iterations
% 		  kernel = {'linear','poly','gauss','hint'} ; 
%		  kervar1, kervar2 = kernel parameters (see 'gram.m' for details)
% Goal  : Learn D and X by min || Phi(Y) - D*X ||_2 , st ||x_i|| <= T0
% Output: Dictionary coefficients A, where D = Phi(Y)*A 
% 	      Sparse coefficients X

Y = params.data; kernel_choice = params.kernel; 
iternum = params.iternum ; 
ds = params.dictsize; 
% init_dic = params.initdic;
kervar1 = params.kervar1; kervar2 = params.kervar2; 

K = gram(Y',Y',kernel_choice, kervar1, kervar2); % compute Gram matrix

mynorm = Knorms(eye(size(K,1)),K) ; 
mynorm = mynorm(:) ;
mynorm = (mynorm*mynorm') ; 
K = K./mynorm ; % normalize to norm-1 in feature space


samplenum = size(Y,2) ; 
D = zeros(ds, samplenum) ; 
label = ceil(ds*rand(1,samplenum));
% if isempty(init_dic)
%     for i=1:ds
%         D(i, label(i)) = 1;  % randomly initilize dictionary
%     end
% else
%     D = init_dic;
% end



last = zeros(1,samplenum);
for it=1:iternum
% while any(label ~= last)
    [~,~,last(:)] = unique(label);   % remove empty clusters
    D = sparse(last,1:samplenum,1, ds, samplenum);
    D = D./sum(D,2);
    T = D*K;
    [~, label] = max(T-dot(T,D,2)/2,[],1);    
end
A = D'; 
X = label';
end

    