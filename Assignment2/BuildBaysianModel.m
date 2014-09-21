
function [model] = BuildBaysianModel(trainData, caseNumber) %crossValidationData, 
% function [model] = BuildBaysianModel(trainData, crossValidationData, caseNumber)
%
% Builds Bayesian model using the given training data and cross validation
% data (optional) for the given case number.
%
% INPUT:
% trainData     : m x n+1 matrix, m is num of examples & n is number of
% dimensions. n+1 th column is for class labels (1 -- for class 1, ... k --
% for class k).
%
% crossValidationData     : (Optional) m x n+1 matrix, m is num of examples & n is
% number of dimensions. n+1 th column is for class labels (1 -- for class
% 1, ... , k -- for class k).
%
% caseNumber: 1 -- Bayes with Covariance same for all classes
%             2 -- Bayes with Covariance different for all classes
%             3 -- Naive Bayes with C = \sigma^2*I
%             4 -- Naive Bayes with C same for all
%             5 -- Naive Bayes with C different for all
%
% OUTPUT:
% model    : k x 2 cell, k is num of classes.
%            Each row i is {muHat(mean_vector)_i, C(covariance_matrix)_i}
%
% See Also : BayesianClassify.m
%

m = size(trainData, 1); % number of training examples
n = size(trainData, 2) - 1; % number of feature dimension
k = length(unique(trainData(:, end))); % number of classes

trainFeatures = trainData(:, 1:end-1);
trainClasses = trainData(:, end);

model = cell(k, 2);
switch caseNumber
    case 1                      % Bayes with Covariance same for all classes
        sigmaHat = zeros(n);
        muHat = cell(k,1);
        for i=1:k
            muHat{i} = mean(trainFeatures(find(trainClasses == i), :));
            for j= transpose(find(trainClasses == i))
                sigmaHat = sigmaHat + transpose(trainFeatures(j, :) - muHat{i})*(trainFeatures(j, :) - muHat{i});
            end
        end
        sigmaHat = sigmaHat/ m;
        for i=1:k
            model{i}{1} = muHat{i};
            model{i}{2} = sigmaHat;
        end
    case 2
        muHat = cell(k,1);
        for i=1:k
            sigmaHat = zeros(n);
            muHat{i} = mean(trainFeatures(find(trainClasses == i), :));
            for j= transpose(find(trainClasses == i))
                sigmaHat = sigmaHat + transpose(trainFeatures(j, :) - muHat{i})*(trainFeatures(j, :) - muHat{i});
            end
            sigmaHat = sigmaHat/size(find(trainClasses == i), 1);
            model{i}{1} = muHat{i};
            model{i}{2} = sigmaHat;
        end
    case 3
        muHat = cell(k,1);
        for i=1:k
            sigmaHat = 0;
            muHat{i} = mean(trainFeatures(find(trainClasses == i), :));
            for j= transpose(find(trainClasses == i))
                sigmaHat = sigmaHat + (trainFeatures(j, :) - muHat{i})*transpose(trainFeatures(j, :) - muHat{i});
            end
            sigmaHat = sigmaHat/size(find(trainClasses == i), 1);
            model{i}{1} = muHat{i};
            model{i}{2} = sigmaHat*eye(n);
        end
    case 4
        sigmaHat = zeros(n);
        muHat = cell(k,1);
        for i=1:k
            muHat{i} = mean(trainFeatures(find(trainClasses == i), :));
            for j= transpose(find(trainClasses == i))
                sigmaHat = sigmaHat + transpose(trainFeatures(j, :) - muHat{i})*(trainFeatures(j, :) - muHat{i});
            end
        end
        sigmaHat = sigmaHat/ m;
        for i=1:k
            model{i}{1} = muHat{i};
            model{i}{2} = diag(diag(sigmaHat));
        end
    case 5
        muHat = cell(k,1);
        for i=1:k
            sigmaHat = zeros(n);
            muHat{i} = mean(trainFeatures(find(trainClasses == i), :));
            for j= transpose(find(trainClasses == i))
                sigmaHat = sigmaHat + transpose(trainFeatures(j, :) - muHat{i})*(trainFeatures(j, :) - muHat{i});
            end
            sigmaHat = sigmaHat/size(find(trainClasses == i), 1);
            model{i}{1} = muHat{i};
            model{i}{2} = diag(diag(sigmaHat));
        end
    otherwise
        disp('Invalid case')
end
end