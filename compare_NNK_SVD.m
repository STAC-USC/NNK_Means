% Compare classification with KSVD, KKSVD, NNK

%% load database
clc; clear; close all;
dataset ='USPS'; % 'MNIST'; %
load(dataset);
results_folder = ['results/', dataset, '/'];
dir_result = mkdir(results_folder);
train_size = size(train_lbl,2);

%% define all paramters - obligatory step
classify_params.run_mode = 'regular';
classify_params.pre_process = 'mean_std';
classify_params.train_images = train_img;
classify_params.train_labels = train_lbl;
classify_params.test_images = test_img;
classify_params.test_labels = test_lbl;
classify_params.num_classes = 10;
classify_params.alg_type = 'KSVD';

classify_params.num_runs = 10;
classify_params.train_per_class = 0;
classify_params.test_per_class = 0;
classify_params.num_atoms = 200;
classify_params.iter = 10; % Number of iterations to run a learning method
classify_params.card = 30; % Max. allowed sparsity of dictionary sparse coding

classify_params.ker_type = 'Gaussian';
classify_params.ker_param_1 = 1;
classify_params.ker_param_2 = 0;

% Parameters for adding noise or removing pixels in the input dataset
classify_params.sigma = 0;
classify_params.missing_pixels = 0;

classify_params.init_dic = 'partial';

%% Check classification accuracy as a function of c=k - approximation dimension
% goal = 'check_num_atoms';
% range = [50 100 200];
% goal = 'check_card';
% range = [5 10 20 30 40 50];
goal = 'check_iter';
range = [5, 10, 50, 100];

%% Algotithms
alg_types = {'KMeans', 'Kernel-KMeans', 'KSVD', 'KKSVD', 'NNK'}; % 
n_methods = length(alg_types);
output_struct = classify_params;
output_struct.alg_type = alg_types;
output_struct.results_mat = cell(n_methods,1);
output_struct.train_images = [];   % deleting data to save space
output_struct.test_images = [];                             
output_struct.train_labels = [];                            
output_struct.test_labels = [];  
output_struct.goal = goal;
output_struct.range = range;

%% Learning loop
for ii = 1:n_methods
    %%
    classify_params.alg_type = alg_types{ii};
    results = classify_main(classify_params,goal, range);
    output_struct.results_mat{ii} = results.results_mat;
end

%% save results
fname = [results_folder, goal, '_result.mat'];
save(fname, 'output_struct');

%% Draw figure
colors = lines(n_methods);
colors(n_methods, :) = [0, 0, 0];
plot_specs = {'x-', '<-',  '^-',  '>-', 'o-','+-','*-', 's-'};

figure
hold on;  grid on;

%% show graphs
% load fname
% x_label_name = 'Sparsity constraint';
x_label_name = 'Number of atoms per class';

% y_label_name = 'Test Accuracy (%)';
% index = 2;
% yticks(0.92:0.01:0.975);
% ylim([0.92, 0.975]);

% y_label_name = 'Train Time (sec)';
% index = 4;

y_label_name = 'Test Time (sec)';
index = 5;
%%

for ii = 1:n_methods
    results = output_struct.results_mat{ii};
    plot(range, results(index, :), plot_specs{ii},'Color', colors(ii,:), 'LineWidth', 1.5, 'DisplayName', alg_types{ii});
    % errorbar(range, output_struct.results_mat{ii}(index, :), ...
%     output_struct.results_mat{ii}(index+1, :), plot_specs{ii}, 'Color', colors(ii,:), 'LineWidth',1.5, 'DisplayName', alg_types(ii)); 
end

xlabel(x_label_name,'FontSize',14,'FontName','Times New Roman');
ylabel(y_label_name,'FontSize',14,'FontName','Times New Roman');

xlim([min(range),max(range)]);
set(gca,'XTick',range);

legend('show'); legend('boxoff') 
legend('Location', 'Best', 'FontName','Times New Roman');
set(gca, 'FontSize', 14);