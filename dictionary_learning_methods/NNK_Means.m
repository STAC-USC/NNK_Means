function [A, X] = NNK_Means(params)
% Input : 
%   	  data = column data; dictsize = number of atoms
%   	  Tdata = sparsity level; iternum = number of iterations
% 		  kernel = {'linear','poly','gauss','hint'} ; 
%		  kervar1, kervar2 = kernel parameters (see 'gram.m' for details)
% Goal  : Learn D and X by min || Phi(Y) - D*X ||_2 , st ||x_i|| <= T0
% Output: Dictionary coefficients A, where D = Phi(Y)*A 
% 	      Sparse coefficients X

Y = params.data; kernel_choice = params.kernel; 
T0 = params.Tdata; iternum = params.iternum ; 
ds = params.dictsize; 
% init_dic = params.initdic;
kervar1 = params.kervar1; kervar2 = params.kervar2; 

K = gram(Y',Y',kernel_choice, kervar1, kervar2); % compute Gram matrix

mynorm = Knorms(eye(size(K,1)),K) ; 
mynorm = mynorm(:) ;
mynorm = (mynorm*mynorm') ; 
K = K./mynorm ; % normalize to norm-1 in feature space


samplenum = size(Y,2) ; 
D = zeros(samplenum, ds) ; 
randid = randperm(samplenum); 

for i=1:ds
    D(randid(i),i) = 1;  % randomly initilize dictionary
end


for it=1:iternum
    [X] = NNK_OMP(D, K, K, T0) ; 
    D = X'/(X*X');
%     X_2 = X.^2;
%     deg = sum(X_2, 2);
%     d = deg;
%     d = d.^(-1);
%     d(deg==0) = 0;
%     D = X'*diag(d);
end

X = NNK_OMP(D, K, K, T0) ;
A = D ; 
end