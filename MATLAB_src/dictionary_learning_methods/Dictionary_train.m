function [train_cell,K_YY_cell,dic_cell,train_t] = Dictionary_train(classify_params)

% ========================================================================
% Modified code from LKDL
% INPUT:
% classify_params - struct containing all parameters for classification.
%
% OUTPUT:
% train_cell      - train set divided to classes
% test_images     - test set
% dic_cell        - trained dictionary divided to classes
% train_t         - total time of training (for entire database)
% ========================================================================

% Parameters in classify_params
train_images          = classify_params.train_images;      % training examples
% test_images           = classify_params.test_images;       % training labels
% test_labels           = classify_params.test_labels;       % test examples
train_labels          = classify_params.train_labels;      % test labels
num_classes           = classify_params.num_classes;       % number of classes in database
alg_type              = classify_params.alg_type;          % algortihm: 'KSVD','KKSVD'
init_dic              = classify_params.init_dic;          % initializtion method of dictionary
num_atoms             = classify_params.num_atoms;         % number of atoms in each class' dictionary
ker_type              = classify_params.ker_type;          % kernel type: 'Gaussian','Polynomial'
ker_param_1           = classify_params.ker_param_1;       % main kernel parameter
ker_param_2           = classify_params.ker_param_2;       % secondary kernel parameter
iter                  = classify_params.iter;              % number of dictionary learning iterations
card                  = classify_params.card;              % cardinality of sparse representations

train_cell = cell(1,num_classes);
K_YY_cell = cell(1,num_classes);

switch alg_type
    case {'kSVD', 'kMeans'}
        kernel_method = false;
    case {'Kernel-kSVD', 'NNK-Means', 'Kernel-kMeans'} % 
        kernel_method = true;
end

% divide the training set to different classes, calculate kernel matrices for each class
for i = 1:num_classes
    train_cell{i} = train_images(:,train_labels(i,:) == 1);
    if kernel_method 
        ker_params = struct('X',0,'Y',0,'ker_type',ker_type,'ker_param_1',ker_param_1,'ker_param_2',ker_param_2);
        ker_params.X = train_cell{i};
        ker_params.Y = train_cell{i};
        YT_Y = train_cell{i}'*train_cell{i};
        K_YY_cell{i} = calc_kernel(YT_Y,ker_params);
    end
end

% initialize dictionary
dic_cell = init_dictionary(classify_params,train_cell);

% dictionary training
train_tic = tic;
h = waitbar(0, 'Training Dictionary');
for i = 1:num_classes
    params = [];
    params.data = train_cell{i};
    params.Tdata = card;
    params.iternum = iter;
    params.dictsize = size(dic_cell{i},2);
    params.memusage = 'high';
    params.kernel = ker_type;
    params.kervar1 = ker_param_2;
    params.kervar2 = ker_param_1;

    switch alg_type
        case 'kSVD'
            [dic_cell{i}, W] = ksvd(params);
        case 'kMeans'
            [dic_cell{i}, W] = KMeans(params);
        case 'Kernel-kMeans'
            [dic_cell{i}, W] = Kernel_KMeans(params);
        case 'Kernel-kSVD'
            [dic_cell{i},W] = KKSVD(params);
        case 'NNK-Means'
            [dic_cell{i},W] = NNK_Means(params);
    end
    waitbar(i/num_classes);
end
close(h);
train_t = toc(train_tic);
end

% dic_cell = train_cell;
% train_t = 0;