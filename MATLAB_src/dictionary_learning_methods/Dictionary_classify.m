function [accuracy, classify_results, sp_codes, classify_t] = Dictionary_classify(classify_params,train_cell,K_YY_cell,test,dic_cell)
% ========================================================================
% Based on the code given by Nguyen et al.
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% INPUT:
% classify_params  - struct containing all parameters for classification.
% train_cell       - virtual train set divided to classes
% test             - virtual test set
% dic_cell         - trained dictionary divided to classes
%
% OUTPUT:
% accuracy         - total accuracy of classification
% classify_results - strurcture containing the calculated labels (to compare with original test labels)
% classify_t       - total time of testing (for entire database)
% ========================================================================

num_classes = classify_params.num_classes;
alg_type = classify_params.alg_type;
ker_type = classify_params.ker_type;
ker_param_1 = classify_params.ker_param_1;
ker_param_2 = classify_params.ker_param_2;

switch alg_type
    case {'kSVD', 'kMeans'}
        kernel_method = false;
    case {'Kernel-kSVD', 'NNK-Means', 'Kernel-kMeans'} % 
        kernel_method = true;
        
end

card = classify_params.card;
X = cell(num_classes,1);
res = zeros(num_classes,size(test,2));
classify_results = zeros(num_classes,size(test,2));
classify_tic = tic;
h = waitbar(0,'Classifying Test Examples');
for i = 1:num_classes
    if kernel_method
        mynorm = Knorms(eye(size(K_YY_cell{i},1)),K_YY_cell{i}) ;
        mynorm = mynorm(:) ;
        K_YY_cell{i} = K_YY_cell{i}./(mynorm*mynorm')  ; % normalize to norm-1 in feature space
        K_ZY = gram(test', train_cell{i}',ker_type,ker_param_2,ker_param_1) ;
        K_ZY = K_ZY./(repmat(mynorm',size(K_ZY,1),1) );
    end
    switch alg_type
        case 'kSVD'
            dict_test_similarity = dic_cell{i}'*test;
            dict_dict_similarity = dic_cell{i}'*dic_cell{i};
%             mask = find_knn_mask(dict_test_similarity', card)';

            X{i} = omp(dict_test_similarity, dict_dict_similarity, card);
            res(i,:) = sqrt(sum((test - dic_cell{i}*X{i}).^2));
        case 'kMeans'
            X1_sum = sum(dic_cell{i}.^2,2);
            X2_sum = sum(test.^2, 1)';
            X1_X2T = test'*dic_cell{i}';
            G = ones(size(test,2),1)*X1_sum' + X2_sum*ones(1, size(dic_cell{i},1)) - 2*X1_X2T;
            [res(i,:), X{i}] = min(G, [], 2);
%              [X{i}, res(i,:)] = dsearchn(dic_cell{i}, test');
        case 'Kernel-kMeans'
            [res(i,:), X{i}] = min(-K_ZY*dic_cell{i}, [], 2);
        case 'Kernel-kSVD'
            K_ZZ = gram(test',test',ker_type,ker_param_2,ker_param_1) ;
            [X{i}, res(i,:)] = KOMP(dic_cell{i},K_YY_cell{i},K_ZY,card,K_ZZ) ;
        case 'NNK-Means'
            K_ZZ = gram(test',test',ker_type,ker_param_2,ker_param_1) ;
            [X{i}, res(i,:)] = NNK_OMP(dic_cell{i},K_YY_cell{i},K_ZY,card,K_ZZ) ;
    end
    waitbar(i/num_classes);
end
close(h);

[~,min_ind] = min(res,[],1);
lin_ind = sub2ind(size(res),min_ind,1:size(res,2));
classify_results(lin_ind) = 1;
diff = sum(abs(classify_results - classify_params.test_labels));
accuracy = sum(diff==0)/length(diff);
% disp([classify_params.alg_type,': ',num2str(accuracy)]);
classify_t = toc(classify_tic);    %% classifcation time
sp_codes = length(find(cat(2, X{:})))/(num_classes*size(test,2)); % Average sparsity of representation in each class for all test examples