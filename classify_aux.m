function [results] = classify_aux(params)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% INPUT:
% classify_params  - struct containing all parameters for classification.
%
% OUTPUT:
% results - struct containing classification results and statistics
% ========================================================================

percent = zeros(params.num_runs,1);
sp_codes = zeros(params.num_runs,1);
train_t = zeros(params.num_runs,1);
classify_t = zeros(params.num_runs,1);
total_t = zeros(params.num_runs,1);
classify_params = params;
virtual_train_t = zeros(params.num_runs,1);
virtual_test_t = zeros(params.num_runs,1);
approximate_dataset = strcmp(params.alg_type, 'LKDL');

for nn = 1:params.num_runs
    
    train = params.train_images;
    train_l = params.train_labels;
    test = params.test_images;
    test_l = params.test_labels;
    
    % adding Gaussian noise with standard deviation: sigma
    if (params.sigma > 0)
        test = test + params.sigma*randn(size(test));
    end
    
    % adding missing pixel corruption
    if (params.missing_pixels > 0)
        for i = 1:size(test,2)
            ind = randperm(size(test,1));
            test(ind(1:ceil(params.missing_pixels*size(test,1))),i) = 0;
        end
    end
    
    % pre-processing
    switch (params.pre_process)
        case 'mean_std'
            train = train - repmat(mean(train),[size(train,1),1]);
            train = train./repmat(sqrt(sum(train.^2)),[size(train,1),1]);
            test = test - repmat(mean(test),[size(test,1),1]);
            test = test./repmat(sqrt(sum(test.^2)),[size(test,1),1]);
        case 'none'
        case 'std'
            train = train./repmat(sqrt(sum(train.^2)),[size(train,1),1]);
            test = test./repmat(sqrt(sum(test.^2)),[size(test,1),1]);
    end
    
    if approximate_dataset
        coreset_size = round(0.15*(size(train,2)));
        ker_params = struct('X',0,'Y',0,'ker_type',params.ker_type,'ker_param_1',params.ker_param_1,'ker_param_2',params.ker_param_2);
        [train, test, virtual_train_t(nn), virtual_test_t(nn)] = calc_virtual_map(train,test,ker_params,'coreset',coreset_size, size(train,1));
        classify_params.alg_type = 'KKSVD';
    end
    
    classify_params.train_images = train;
    classify_params.train_labels = train_l;
    classify_params.test_images = test;
    classify_params.test_labels = test_l;
    
    total_tic = tic;
    [train_cell,K_YY_cell,test,dic_cell,train_t(nn)] = Dictionary_train(classify_params);
    [percent(nn), classify_results, sp_codes(nn), classify_t(nn)] = Dictionary_classify(classify_params,train_cell,K_YY_cell,test,dic_cell);
    total_t(nn) = toc(total_tic);
    disp([num2str(nn),' out of ',num2str(params.num_runs), ', Accuracy: ',num2str(percent(nn)), ', sig dim: ',num2str(size(train,1)), ', Avg. sparsity: ', num2str(sp_codes(nn))]);
end

results.percent = mean(percent);                   % averaged accuracy result of classification
results.std = std(percent);                        % std of accuracy results over num_runs

results.train_t = mean(train_t + virtual_train_t);                   % total training time for entire database
results.classify_t = mean(classify_t + virtual_test_t);             % total test time for entire database
results.total_t = mean(total_t);                   % toal runtime: train_time + test_time + other_time(virtual samples)
results.class_vec = classify_results;              % resulting labels after classification
results.sp_codes = mean(sp_codes);                       % sparsity of codes of test samples

disp(['Average accuracy: ',num2str(results.percent),', std: ',num2str(results.std), ', Avg. sparsity: ', num2str(results.sp_codes)]);