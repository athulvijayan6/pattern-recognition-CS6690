function [classLabels] = BayesianClassify(model, testData, prior)
% function [classLabels] = BayesianClassify(model, testData)
%
% Gives the class labels of testData according to the given model
%
% INPUT:
%
% model    : k x 2 cell, k is num of classes.
%            Each row i is {muHat(mean_vector)_i, C(covariance_matrix)_i}
%
% testData;     : m x n matrix, m is num of examples & n is
% number of dimensions.
%
% OUTPUT:
%
% classLabels: m x 1 matrix, labels of testData, 1 for class 1, ... , k for
% class k.
%
% See Also : BuildBaysianModel.m
%

m = size(testData, 1); % number of examples
n = size(testData, 2); % number of feature dimension
k = size(model, 1); % number of classes

if nargin < 3
    prior = (1/k)*ones(k, 1);
end

classLabels  = zeros(m,1);
for i=1:m
    x = testData(i, :);
    for j=1:k
        muHat = model{j}{1};
        sigmaHat = model{j}{2};
        g(j) = -(1/2)*(x - muHat)*inv(sigmaHat)*transpose(x - muHat) -(1/2)*log(det(sigmaHat)) + log(prior(k));
    end
    classLabels(i) = find(g == max(g));
end