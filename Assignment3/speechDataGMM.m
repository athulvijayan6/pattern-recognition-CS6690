% @Author: athul
% @Date:   2014-10-21 16:04:10
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-23 11:48:13

clear all;
clc;
warning off;
% 4 7 9 2 5

DO_TRAINING = true;
dumpFile = 'speechGMM_n10.mat';

if DO_TRAINING
    paths = {'digit_data/four/' ; 'digit_data/seven/'; 'digit_data/nine/'; 'digit_data/two/'; 'digit_data/five/'};
    trainData = zeros(1, 40);
    numNorms = [10 10 10 10 10];
    for i=1: size(paths, 1)
        allFiles = struct2table(dir(strcat(paths{i}, '*.txt')));
        allFiles = table2array(allFiles(:, 1));
        % allFiles = allFiles(1:15, :);
        allFiles = allFiles(randperm(size(allFiles, 1)), :);
        trainFiles{i} = allFiles(1:floor(0.8*length(allFiles)), :);
        testFiles{i} = setdiff(allFiles, trainFiles{i});
        for j=1:length(trainFiles{i})
            d = importdata(strcat(paths{i}, trainFiles{i}{j}), ' ');
            d(:, end + 1) = i*ones(length(d), 1);
            trainData = vertcat(trainData, d);
        end
    end
    trainData(1, :) = [];
    [model, likelihood] = GMMtrain(trainData, numNorms, 'diagonal', 50, 1e-2);
    clearvars -except model allFiles trainFiles testFiles trainData numNorms dumpFile paths likelihood;
    disp(strcat('Done training ...... saving model to  ', dumpFile))
    save(dumpFile);
else
    load(dumpFile);
    disp(strcat('Done loading model from ', dumpFile));
end

% we make a result matrix having (scores, predicted label, actual label) as columns
result = zeros(1, length(testFiles) + 2);
for i=1:size(testFiles, 2)
    for j=1:length(testFiles{i})
        d = importdata(strcat(paths{i}, testFiles{i}{j}), ' ');
        [~, s] = GMMclassify(model, d);
        log_s = zeros(1, size(s, 2));
        for m=1:length(s)
            log_s = log_s + log(s(m, :));
        end
        predLabel = find(log_s == max(log_s));
        result = vertcat(result, [log_s predLabel i]);
    end
end
result(1, :) = [];



        





