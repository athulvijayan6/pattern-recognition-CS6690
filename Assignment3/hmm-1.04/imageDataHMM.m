% @Author: Athul Vijayan
% @Date:   2014-10-23 15:25:44
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-11-13 20:05:43
clear('all');
clc

DO_TRAINING = false;

if DO_TRAINING
    if ~exist('image/imageDataExtracted.mat')
        paths = {'../highway/features/' ; '../insidecity/features/'; '../mountain/features/'};
        trainData = zeros(1, 25);
        testData = zeros(1, 25);
        for i=1: size(paths, 1)
            allFiles = struct2table(dir(strcat(paths{i}, '*.jpg_color_edh_entropy')));
            allFiles = table2array(allFiles(:, 1));
            allFiles = allFiles(randperm(size(allFiles, 1)), :);
            trainFiles{i} = allFiles(1:floor(0.8*length(allFiles)), :);
            testFiles{i} = setdiff(allFiles, trainFiles{i});
            for j=1:length(trainFiles{i})
                d = importdata(strcat(paths{i}, trainFiles{i}{j}), ' ');
		d = zscore(d);
                d(:, end + 1) = i*ones(length(d), 1);
                d(:, end + 1) = j*ones(length(d), 1);
                trainData = vertcat(trainData, d);
            end

            for j=1:length(testFiles{i})
                d = importdata(strcat(paths{i}, testFiles{i}{j}), ' ');
		d = zscore(d);
                d(:, end + 1) = i*ones(length(d), 1);
                d(:, end + 1) = j*ones(length(d), 1);
                testData = vertcat(testData, d);
            end
        end
        trainData(1, :) = [];   testData(1, :) = [];
        clear('d', 'i', 'j');
        save('image/imageDataExtracted.mat');
        disp('saving extracted data...');
    else
        load('image/imageDataExtracted.mat');
        disp('loaded extracted data...')
    end

    disp('starting training...');
    dumpFile = 'image/imageHMM_s6k256.mat';
    numSyms = 256;
    numStates = 24;
    numClasses = max(trainData(:, end-1));
    [labels, centroids] = kmeans(trainData(:,1:end-2), numSyms, 'MaxIter',1000);
    trainSeqFiles = {'image/img_train_s6k256_c1.hmm.seq','image/img_train_s6k256_c2.hmm.seq', 'image/img_train_s6k256_c3.hmm.seq'};
    for i=1:numClasses
        if exist(trainSeqFiles{i})
            delete(trainSeqFiles{i});
        end
        classData = trainData(find(trainData(:, end-1) == i), end);
        classLabels = labels(find(trainData(:, end-1) == i));
        for j=1:max(classData)
            seq = classLabels(find(classData == j));
            seq = seq' - 1;
            dlmwrite(trainSeqFiles{i}, seq, '-append', 'delimiter', ' ');
        end
        disp(['Written train sequence file to ', trainSeqFiles{i}]);
    end
    disp('Done writing training sequences...')
    clear('labels');
    testSeqFiles = {'image/img_test_s6k256_c1.hmm.seq','image/img_test_s6k256_c2.hmm.seq', 'image/img_test_s6k256_c3.hmm.seq'};    
    for i=1:numClasses
        if exist(testSeqFiles{i})
            delete(testSeqFiles{i});
        end
        classData = testData(find(testData(:, end-1) == i), :);
        for j=1:max(classData(:, end))            
            letter = classData(find(classData(:, end) == j), 1:end-2);
            for p=1:size(letter, 1)
                for m=1:size(centroids, 1)
                    d(m) = norm(letter(p, :) - centroids(m, :));
                end
                seq(p) = find(d == min(d)) - 1;
            end
            dlmwrite(testSeqFiles{i}, seq, '-append', 'delimiter', ' ');
        end
        disp(['Written test sequence file to ', testSeqFiles{i}]);
    end
    disp('Done writing testing sequences...')
    save(dumpFile);
else
    dumpFile = 'image/imageHMM_s6k256.mat';
    load(dumpFile);
    disp(['Done loading model from ', dumpFile]);
end


MAKE_INIT = false;
initProbabs = {'image/img_train_s6k256_c1.hmm','image/img_train_s6k256_c2.hmm', 'image/img_train_s6k256_c3.hmm'};
if MAKE_INIT
    for i=1: length(initProbabs)
        % scale = 4;
        f = fopen(initProbabs{i}, 'w');
        fprintf(f, 'states: %d\n', numStates);
        fprintf(f, 'symbols: %d\n\n', numSyms);
        fclose(f);
        for j=1:numStates
            m = numSyms/numStates;
            % p = 1/(numSyms + m*(scale-1));
            p = 1/numSyms;
            prob = p*ones(1, numSyms);
            % prob((j-1)*m+1:j*m) = scale*p*ones(1, m);
            if j<numStates
                dlmwrite(initProbabs{i}, [0.5 prob], '-append', 'delimiter', ' ');
                dlmwrite(initProbabs{i}, [0.5 prob], '-append', 'delimiter', ' ');
            else
                dlmwrite(initProbabs{i}, [1 prob], '-append', 'delimiter', ' ');
                dlmwrite(initProbabs{i}, zeros(1, numSyms), '-append', 'delimiter', ' ');
            end
            f = fopen(initProbabs{i}, 'a');
            fprintf(f, '\n');
            fclose(f);                
        end
    end
end
DO_HMM_TRAINING = false;
if DO_HMM_TRAINING
    for i=1:length(trainSeqFiles)
        [status, output] = unix(['./train_hmm ', trainSeqFiles{i}, ' ', initProbabs{i}, ' .01']);
        if (status ~= 0)
            disp('HMM train code did not run..');
            disp(output);
            break;
        end
        disp(output);            
    end
    disp('done training..')
end

modelFiles = {'image/img_train_s6k256_c1.hmm.seq.hmm','image/img_train_s6k256_c2.hmm.seq.hmm', 'image/img_train_s6k256_c3.hmm.seq.hmm'};
result = zeros(1, length(modelFiles) + 2);
for i=1:length(testSeqFiles)
    for j=1:length(modelFiles)
        [status, output] = unix(['./test_hmm ', testSeqFiles{i}, ' ', modelFiles{j}]);
        if (status ~= 0)
            disp('HMM test code did not run..');
            disp(output);
            break;
        end
        disp(output);
        s = importdata('alphaout', ' ')';
        scores(:, j) = s;
    end
    for m=1:size(scores, 1)
        % scores(m, :) = scores(m, :)./sum(scores(m, :));
        l(m) = find(scores(m, :) == max(scores(m, :)));
    end
    result = vertcat(result, [scores l' i*ones(size(scores, 1), 1)]);
    clear('i', 'j', 'status', 'output', 's', 'scores', 'labels');
end
result(1, :) = [];

% <<======================== Performace metrics starts ============================>>
trueClass=result(:,end);
predClass=result(:, end-1);

[C,or]= confusionmat(trueClass, predClass);

printmat(C, 'Confution Matrix', 'ActCLASS1 CLASS2 CLASS3', 'PredCLASS1 CLASS2 CLASS3' );
Accuracy=(sum(diag(C)))/(sum(sum(C)))*100;
disp('ACCURACY(%)=');disp(Accuracy);

k=numClasses;
D=C;D(1:k+1:k*k) = 0;

for i=1:k
    Pclass(i)=C(i,i)/sum(C(i,:));
    IError(i)=sum(D(i,:))/sum(C(i,:));
    EError(i)=sum(D(:,i))/sum(C(:,i));
end

PE=horzcat(Pclass',IError',EError');
printmat(PE, 'Precision Error', 'CLASS1 CLASS2 CLASS3', 'Precision inclusionEr exclusionEr' );
scores = result(:, 1:end-2);
for i=1:k
    z(:,i)=((scores(:,i))-mean(scores(:,i)))/std(scores(:,i));
end

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
title('DET curve for spiral data');

%ROC Curve
Tscore=vertcat(targetScores,nonTargetScores);
NLabel1(1:length(targetScores))=1;
NLabel2(1:length(nonTargetScores))=0;
NLabel=vertcat(NLabel1',NLabel2');
[X1,Y1] = perfcurve(NLabel,Tscore,1);
figure;
plot(X1,Y1);
title('ROC spiral data');
xlabel('False Positive rate');ylabel('True positive rate');

% <<======================== Performace metrics ends ============================>>
