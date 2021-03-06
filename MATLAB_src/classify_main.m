function results_struct = classify_main(classify_params,parameter,values)
% ========================================================================
% Modified code from LKDL 
% ========================================================================
% 
% INPUT:
% [classify_params] - struct contains all params needed for classification
% [parameter]       - string containing parameter needed checking
% [values]          - values of the parameter
% example: results = classify_main(class_params,'check noise sigma',[0:0.1:2]);
%                    This performs training and classification in the
%                    presence of noise with random Gaussian noise with 
%                    0:0.1:2 sigma values.
% example: results = classify_main(class_params,'check noise sigma',0);
%                    This runs the chosen algorithm with no parameter
%                    changes. The most basic use of 'classify_main'.
% OUTPUT:
% [results_struct]  - same as classify_params struct, containing all params,
%                     along with results: accuracy, training, test and total time.

results_mat = zeros(5,length(values));

for i = 1:length(values)
    switch (parameter)
        case ('check_num_atoms')                           % check number of atoms in dictionary learning
            classify_params.num_atoms = values(i);
        case ('check_ker_param_1')                         % check main kernel parameter
            classify_params.ker_param_1 = values(i);
        case ('check_ker_param_2')                         % check secondary kernel parameter
            classify_params.ker_param_2 = values(i);      
        case ('check_noise_sigma')                         % add noise level to test images to currupt them
            classify_params.sigma = values(i); 
        case ('check_missing_pixels')                      % corrupt test images by zeroing random pixels
            classify_params.missing_pixels = values(i);
        case ('check_iter')                                % check number of iteration of dictionary learning
            classify_params.iter = values(i);
        case ('check_card')                                % check cardinality in atom-decomposition/sparse coding
            classify_params.card = values(i);
    end
    
    [~,space] = regexp(parameter,'check_');

    disp(['Checking ',parameter(space+1:end),' = ',num2str(values(i)),', ',classify_params.alg_type,', '...
    num2str(i),' out of ',num2str(length(values)),' runs']);

    
    % separate auxiliary function for mini-batch dictionary learning
    results = classify_aux(classify_params);

   
    results.value = values(i);
    results_mat(1,i) = results.value;                            % value of current parameter checked
    results_mat(2,i) = results.percent;                          % accuracy result of classification
    results_mat(3,i) = results.std;                              % std of accuracy in case of more than 1 run
    results_mat(4,i) = results.train_t;                          % total training time for entire database
    results_mat(5,i) = results.classify_t;                       % total test time for entire database
    results_mat(6,i) = results.total_t;                          % total runtime: train_time + test_time + other_time
    
%     results_struct = classify_params;                            % save classification params used in experiment
    results_struct.results_mat = results_mat;                    % results_matrix
%     results_struct.train_images = [];                            % deleting train_images (save space)
%     results_struct.test_images = [];                             % deleting train_labels (save space)
%     results_struct.train_labels = [];                            % deleting test_images  (save space)
%     results_struct.test_labels = [];                             % deleting test_labels  (save space)
    results_struct.class_vec = results.class_vec;                % resulting labels after classification
    results_struct.sp_codes = results.sp_codes;                  % sparse codes of test samples
end