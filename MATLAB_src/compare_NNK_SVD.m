% Compare classification with KSVD, KKSVD, NNK

%% load database
clc; clear; close all;
dataset = 'USPS'; % 'cifar10'; % 'MNIST'; %  'simulated_data'; %  
load(dataset);

%%
results_folder = ['results/', dataset, '/'];
dir_result = mkdir(results_folder);
train_size = size(train_img,2);
num_classes = size(train_lbl,1);
%% define all paramters - defaults
classify_params.run_mode = 'regular';
classify_params.pre_process = 'mean_std'; %'none'; %
classify_params.train_images = double(train_img);
classify_params.train_labels = double(train_lbl);
classify_params.test_images = double(test_img);
classify_params.test_labels = double(test_lbl);
classify_params.num_classes = num_classes;
classify_params.alg_type = 'KSVD';

classify_params.train_size = train_size; % round(0.2*train_size); % 
classify_params.num_atoms = 50; % 15; %
classify_params.num_runs = 1; % 1;%
classify_params.iter = 10; % Number of iterations to run a learning method
classify_params.card =  30; % 5; %  % Max. allowed sparsity of dictionary sparse coding

classify_params.ker_type = 'Gaussian';
classify_params.ker_param_1 = 1;
classify_params.ker_param_2 = 0;

% Parameters for adding noise or removing pixels in the input dataset
classify_params.sigma = 0;
classify_params.missing_pixels = 0;

classify_params.init_dic = 'partial';

%% Check classification accuracy for different parameter variation
goal = 'check_classification';
range = [classify_params.card];

% goal = 'check_noise_sigma';
% range = 0:0.5:2 ;
% x_label_name = 'Noise level';

% goal = 'check_missing_pixels';
% range = 0:0.2:0.9;
% x_label_name = '% Missing pixels';

% goal = 'check_num_atoms';
% range = [50 100 200];
% x_label_name = 'Number of atoms per class';

% goal = 'check_card';
% range = [5 10 20 30 40 50];
% x_label_name = 'Sparsity constraint';

% goal = 'check_iter';
% range = [5, 10, 50, 100];



%% Algotithms
alg_types = {'kMeans', 'Kernel-kMeans', 'kSVD', 'Kernel-kSVD', 'NNK-Means'};
n_methods = length(alg_types);
output_struct = classify_params;
output_struct.alg_type = alg_types;
output_struct.results = cell(n_methods,1);
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
    output_struct.results{ii} = classify_main(classify_params,goal, range);
end

%% save results
fname = [results_folder, goal, '_result.mat'];
save(fname, 'output_struct');

%% Draw figure
alg_types = output_struct.alg_type;
range = output_struct.range;
n_methods = length(alg_types);
colors = lines(n_methods);
colors(n_methods, :) = [0, 0, 0];
plot_specs = {'x-', '<-',  '^-',  '>-', 'o-','+-','*-', 's-'};

figure
hold on;  grid on;

% show graphs
% load fname
% y_label_name = 'Test Accuracy (%)';
% index = 2;
% yticks(0.92:0.01:0.97);
% ylim([0.92, 0.97]);

% y_label_name = 'Train Time (sec)';
% index = 4;

y_label_name = 'Test Time (sec)';
index = 5;
% Plot for check case

for ii = 1:n_methods
    results = output_struct.results{ii}.results_mat;
    plot(range, results(index, :), plot_specs{ii},'Color', colors(ii,:), 'LineWidth', 1.5, 'DisplayName', alg_types{ii});
    % errorbar(range, output_struct.results{ii}.results_mat(index, :), ...
%     output_struct.results{ii}.results_mat(index+1, :), plot_specs{ii}, 'Color', colors(ii,:), 'LineWidth',1.5, 'DisplayName', alg_types(ii)); 
end

xlabel(x_label_name,'FontSize',14,'FontName','Times New Roman');
ylabel(y_label_name,'FontSize',14,'FontName','Times New Roman');

xlim([min(range),max(range)]);
set(gca,'XTick',range);

legend('show'); legend('boxoff') 
legend('Location', 'Best', 'FontName','Times New Roman', 'FontSize', 20);
% set(gca, 'FontSize', 14);

%% Visualization plot
% for ii = 1:n_methods
%     %%
%   figure;
%   hold on; axis off;
%   col = lines(4);% {[1 0 0], [0.6 0 0], [0.3 0 0], [0 1 1]};
%   lbl = output_struct.results{ii}.class_vec;
% %   for i = 1:classify_params.num_classes
% %     plot(test_img(1,find(lbl(i, :))),test_img(2,find(lbl(i, :))),'o','Color', col(i,:),'markersize',7, 'MarkerFaceColor', col(i,:));
% %   end
%   colormap(col);
%   [l, ~] = find(lbl);
%   [~, h] = contourf(xx, yy, reshape(l, 101,101), 1:classify_params.num_classes, 'LineColor', 'none');
%   xlim([-1, 1]);
%   ylim([-1, 1]);
%   set(gca,'XTick',[-1, -0.5, 0, 0.5, 1], 'YTick', [-1, -0.5, 0, 0.5, 1]);
%   hFills = h.FacePrims;
%   [hFills.ColorType] = deal('truecoloralpha');
%   for idx = 1:numel(hFills)
%     hFills(idx).ColorData(4) = 128;
%   end
% %   title([alg_types{ii} ' - Accuracy: ' num2str(output_struct.results{ii}.results_mat(2,:)*100) '%']);
% end