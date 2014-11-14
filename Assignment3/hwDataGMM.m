% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-11-13 12:37:18

clear all;
clc;
warning off;
disp('hand writing data GMM..')
DO_TRAINING = false;
if DO_TRAINING
    if ~exist('hwDataExtracted.mat')
        path = 'FeaturesHW/';
        files = dir(strcat(path, '*.ldf'));
        files = {files.name};
        trainData = zeros(1, 4);
        testData = zeros(1, 4);
        for i=1: length(files)
            file = fopen(strcat(path, files{i}));
            tline = fgets(file);
            numPt = 1;
            while ischar(tline)
                d{numPt} = str2num(tline);
                d{numPt} = d{numPt}(2:end);
                d{numPt} = reshape(d{numPt}, 2, [])';
                d{numPt} = zscore(d{numPt});
                numPt = numPt + 1;
                tline = fgets(file);
            end
            disp(['Parsed file ', file]);
            trainArray = d(1:0.7*length(d));
            testArray = d(0.7*length(d):end);
            for j = 1:length(trainArray)
                labels = ones(length(trainArray{j}), 1);
                trainData = vertcat(trainData, [trainArray{j} i*labels j*labels]);
            end
            clear('labels', 'j')
            for j = 1:length(testArray)
                labels = ones(length(testArray{j}), 1);
                testData = vertcat(testData, [testArray{j} i*labels j*labels]);
            end

            trainData(1, :) =[]; testData(1, :) =[];
        end
        clearvars -except trainData testData DO_TRAINING path trainArray testArray files;
        save('hwDataExtracted.mat');
        disp('saved extracted data to hwDataExtracted.mat');
    else
        load('hwDataExtracted.mat');
        disp('Loaded hwDataExtracted.mat.....');
    end
    numNorms = [3 3 3 3 3 3 3 3];
    dumpFile = 'hwDataGMM_n24.mat';

    disp('starting training...');
    [model, likelihood] = GMMtrain(trainData(:, 1:end-1), numNorms, 'diagonal', 40, 1e-1);
    clearvars -except model files trainData testData DO_TRAINING path trainArray testArray files numNorms dumpFile likelihood;
    disp(strcat('Done training ...... saving model to  ', dumpFile))
    save(dumpFile);
else
    dumpFile = 'hwDataGMM_n3.mat';
    load(dumpFile);
    disp(strcat('Done loading model from ', dumpFile));
end

% we make a result matrix having (scores, predicted label, actual label) as columns
result = zeros(1, length(files) + 2);
for i=1:length(files)
    classData = testData(find(testData(:, end-1) == i), :);
    for j=1:max(classData(:, end))
        d = classData(find(classData(:, end) == j), 1:end-2);
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

printmat(C, 'Confution Matrix', 'ActCLASS1 CLASS2 CLASS3 CLASS4 CLASS5 CLASS6 CLASS7 CLASS8', 'PredCLASS1 CLASS2 CLASS3 CLASS4 CLASS5 CLASS6 CLASS7 CLASS8' );
Accuracy=(sum(diag(C)))/(sum(sum(C)))*100;
disp('ACCURACY(%)=');disp(Accuracy);

k=8;
D=C;D(1:k+1:k*k) = 0;

for i=1:k
    Pclass(i)=C(i,i)/sum(C(i,:));
    IError(i)=sum(D(i,:))/sum(C(i,:));
    EError(i)=sum(D(:,i))/sum(C(:,i));
end

PE=horzcat(Pclass',IError',EError');
printmat(PE, 'Precision Error', 'CLASS1 CLASS2 CLASS3 CLASS4 CLASS5 CLASS6 CLASS7 CLASS8', 'Precision inclusionEr exclusionEr' );
scores = result(:, 1:end-2);
scores = exp(scores);
for m=1:length(scores)
    scores(m, :) = scores(m, :)./sum(scores(m, :));
end
z = zscore(scores);

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
