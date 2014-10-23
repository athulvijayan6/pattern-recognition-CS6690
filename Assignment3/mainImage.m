% @Author: Athul Vijayan
% @Date:   2014-10-16 16:42:30
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-23 23:07:06

% Load the feature data into matrices
% @Author: athul
% @Date:   2014-10-21 16:04:10
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-22 22:38:19

clear all;
clc;
warning off;
% 4 7 9 2 5

DO_TRAINING = true;
dumpFile = 'imageGMM_n15.mat';

if DO_TRAINING
    paths = {'highway/features/' ; 'insidecity/features/'; 'mountain/features/'};
    trainData = zeros(1, 24);
    numNorms = [15 15 15];
    for i=1: size(paths, 1)
        allFiles = struct2table(dir(strcat(paths{i}, '*.jpg_color_edh_entropy')));
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
    disp('starting training...');
    [model, likelihood] = GMMtrain(trainData, numNorms, 'diagonal', 20, 100);
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



        





