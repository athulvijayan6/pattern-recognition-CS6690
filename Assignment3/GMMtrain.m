% @Author: Athul Vijayan
% @Date:   2014-10-18 18:09:28
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-23 07:10:41

%% GMMtrain: function description
function [model, likelihood] = GMMtrain(trainData, numNorms, cov_type, max_iter, delta_cutoff)

    if nargin ==2
        cov_type = 'diagonal';
        max_iter = 100;
        delta_cutoff = 1e-5;
    elseif nargin == 3
        max_iter = 100;
        delta_cutoff = 1e-5;
    elseif nargin == 4
        delta_cutoff = 1e-5;
    end

    m = size(trainData, 1); % number of training examples
    n = size(trainData, 2) - 1; % number of feature dimension
    k = length(unique(trainData(:, end))); % number of classes

    model = cell(k, 3);

    trainFeatures = trainData(:, 1:end-1);
    trainClasses = trainData(:, end);
    likelihood = cell(k, 1);
    for i=1:k
        [muHat, sigmaHat, piHat, likelihood{i}] = GMMpdf(trainFeatures(find(trainClasses == i), :), numNorms(i), cov_type, max_iter, delta_cutoff);
        disp(['class ', num2str(i), ' training done']);
        model{i}{1} = muHat;
        model{i}{2} = sigmaHat;
        model{i}{3} = piHat;
    end
end

