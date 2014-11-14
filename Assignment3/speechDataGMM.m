% @Author: athul
% @Date:   2014-10-21 16:04:10
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-11-08 19:22:53

clear all;
clc;
warning off;
% 4 7 9 2 5

DO_TRAINING = false;
disp('Speech data GMM...');
if DO_TRAINING
    paths = {'digit_data/four/' ; 'digit_data/seven/'; 'digit_data/nine/'; 'digit_data/two/'; 'digit_data/five/'};
    trainData = zeros(1, 40);
    if ~exist('speechDataExtracted.mat')
        for i=1: size(paths, 1)
            allFiles = struct2table(dir(strcat(paths{i}, '*.txt')));
            allFiles = table2array(allFiles(:, 1));
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
        clear('d', 'i', 'j');
        save('speechDataExtracted.mat');
        disp('saving extracted data...');
    else
        load('speechDataExtracted.mat');
        disp('loaded extracted data...')
    end

    dumpFile = 'speechGMM_n8.mat';
    numNorms = [8 8 8 6 8];
    disp('starting training...')
    [model, likelihood] = GMMtrain(trainData, numNorms, 'diagonal', 15, 100);
    clearvars -except model allFiles trainFiles testFiles trainData numNorms dumpFile paths likelihood;
    disp(strcat('Done training ...... saving model to  ', dumpFile))
    save(dumpFile);
else
    dumpFile = 'speechGMM_n3.mat';
    load(dumpFile);
    disp(strcat('Done loading model from ', dumpFile));
end

% we make a result matrix having (scores, predicted label, actual label) as columns
result = zeros(1, length(testFiles) + 2);
for i=1:size(testFiles, 2)
    for j=1:length(testFiles{i})
        d = importdata(strcat(paths{i}, testFiles{i}{j}), ' ');
        [~, s] = GMMclassify(model, d);
        scores = sum(log(s));
        predLabel = find(scores == max(scores));
        result = vertcat(result, [scores predLabel i]);
    end
end
result(1, :) = [];

% <<======================== Performace metrics starts ============================>>
clear('scores');
trueClass=result(:,end);
predClass=result(:, end-1);

[C,or]= confusionmat(trueClass, predClass);

printmat(C, 'Confution Matrix', 'ActCLASS1 CLASS2 CLASS3 CLASS4 CLASS5', 'PredCLASS1 CLASS2 CLASS3 CLASS4 CLASS5' );
Accuracy=(sum(diag(C)))/(sum(sum(C)))*100;
disp('ACCURACY(%)=');disp(Accuracy);

k=5;
D=C;D(1:k+1:k*k) = 0;

for i=1:k
    Pclass(i)=C(i,i)/sum(C(i,:));
    IError(i)=sum(D(i,:))/sum(C(i,:));
    EError(i)=sum(D(:,i))/sum(C(:,i));
end

PE=horzcat(Pclass',IError',EError');
printmat(PE, 'Precision Error', 'CLASS1 CLASS2 CLASS3 CLASS4 CLASS5', 'Precision inclusionEr exclusionEr' );
scores = result(:, 1:end-2);
for m=1:length(scores)
    scores(m, :) = scores(m, :)./sum(scores(m, :));
end
z = -1*zscore(scores);

targetScores = 0;
for i=1:k
    targetScores = vertcat(targetScores, z(find(trueClass(:, 1) == i), i));
end
targetScores(1) = [];
nonTargetScores = setdiff(reshape(z, [], 1), targetScores);
figure;
[f1, g1]=Compute_DET(targetScores, nonTargetScores);
Plot_DET(f1,g1, 'r');
set(get(gca,'XLabel'),'String','False positive rate');
set(get(gca,'YLabel'),'String','Miss detection rate');
title('DET curve for speech data');

%ROC Curve
Tscore=vertcat(targetScores,nonTargetScores);
NLabel1(1:length(targetScores))=1;
NLabel2(1:length(nonTargetScores))=0;
NLabel=vertcat(NLabel1',NLabel2');
[X1,Y1] = perfcurve(NLabel,Tscore,1);
figure;
plot(X1,Y1);
title('ROC speech data');
% <<======================== Performace metrics ends ============================>>
