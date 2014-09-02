% @Author: athul
% @Date:   2014-08-28 13:51:17
% @Last Modified by:   athul
% @Last Modified time: 2014-08-29 16:23:13

% choose data input
clear all
inputFile = 'Data_2_r0_s29.txt';

inData = dlmread(inputFile, ' ');  % delimiter is space.
X = [ones(size(inData, 1), 1) inData(:, 1:size(inData, 2)-1)];
y = inData(:, size(inData, 2));

% Now we have two evaluation scheme:

% 1. Leave-p-out cross-validation: take 2/3 rd data for training, 1/3rd for validation
% 2. Leave one sample out evaluation
for i=1:round(size(X,1)/3)
    trainIds = [i: (i-1) + round(2*size(X,1)/3)];
    valIds = setxor([1:size(X,1)], intersect(trainIds, [1:size(X,1)]));
    Xtrain = X(trainIds, :);
    yTrain = y(trainIds, :);
    Xval = X(valIds, :);
    yVal = y(valIds, :);

    theta(i, :) = transpose(inv(Xtrain'*Xtrain)*Xtrain'*yTrain);
    % Now evaluate for validation set
    yhat = Xval*theta(i,:)';
    rmse(i) = rms(yVal - yhat);
end
[c, index] = min(rmse); clear('c');
disp(['best model is for theta using leave p out validation is ',num2str(theta(index,:))]);

% Leave one sample out evaluation

for i=1:size(X, 1)
    Xtrain = removerows(X, i);
    yTrain = removerows(y, i);
    Xval = X(i, :);
    yVal = y(i, :);

    theta2(i,:) = transpose(inv(Xtrain'*Xtrain)*Xtrain'*yTrain);
    yhat = Xval*theta2(i,:)';
    rmse(i) = rms(yVal - yhat);
end

% Minimum error; i.e best model is obtained for least rmse.
[c, index] = min(rmse); clear('c');
disp(['best model is for theta ',num2str(theta2(index,:))]);


