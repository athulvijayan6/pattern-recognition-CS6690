% @Author: Athul Vijayan
% @Date:   2014-10-23 15:25:44
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-11-14 13:35:23
clear('all');
clc

DO_TRAINING = true;

if DO_TRAINING
    if ~exist('hw/hwDataExtracted.mat')
        path = '../FeaturesHW/';
        files = dir(strcat(path, '*.ldf'));
        files = {files.name};
        trainData = zeros(1, 8);
        testData = zeros(1, 8);
        for i=1: length(files)
            file = fopen(strcat(path, files{i}));
            tline = fgets(file);
            numPt = 1;
            while ischar(tline)
                d{numPt} = str2num(tline);
                d{numPt} = d{numPt}(2:end);
                d{numPt} = reshape(d{numPt}, 2, [])';
                d{numPt} = zscore(d{numPt});
                for pt =2: size(d{numPt}, 1)-1
                    d_dot(pt -1, 1) = (d{numPt}(pt+1, 1) - d{numPt}(pt-1, 1));
                    d_dot(pt -1, 2) = (d{numPt}(pt+1, 2) - d{numPt}(pt-1, 2));
                    d_ddot(pt -1, 1) = (d{numPt}(pt+1, 1) - 2*d{numPt}(pt, 1) + d{numPt}(pt-1, 1));
                    d_ddot(pt -1, 2) = (d{numPt}(pt+1, 2) - 2*d{numPt}(pt, 2) + d{numPt}(pt-1, 2));
                end
                d_dot = [d_dot(1, :) ; d_dot];
                d_ddot = [d_ddot(1, :) ; d_ddot];
                d_dot(end +1, :) = d_dot(end, :);
                d_ddot(end +1, :) = d_ddot(end, :);
                d{numPt} = horzcat(d{numPt}, d_dot);
                d{numPt} = horzcat(d{numPt}, d_ddot);
                numPt = numPt + 1;
                tline = fgets(file);
                clear('d_dot');
            end
            disp(['Parsed file ', file]);
            m = floor(0.7*length(d));
            trainArray = d(1:m);
            testArray = d(m:end); clear('m');
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
        save('hw/hwDataExtracted.mat');
        disp('saved extracted data to hwDataExtracted.mat');
    else
        load('hw/hwDataExtracted.mat');
        disp('Loaded hwDataExtracted.mat.....');
    end

    disp('starting training...');
    numSyms = 128 
    numStates = 6 
    numClasses = max(trainData(:, end-1));
    dumpFile = 'hw/hwHMM_s6k128.mat';
    [labels, centroids] = kmeans(trainData(:,1:end-2), numSyms, 'MaxIter',1000);
    trainSeqFiles = {'hw/hw_train_s6k128_c1.hmm.seq', 'hw/hw_train_s6k128_c2.hmm.seq', 'hw/hw_train_s6k128_c3.hmm.seq', 'hw/hw_train_s6k128_c4.hmm.seq', 'hw/hw_train_s6k128_c5.hmm.seq', 'hw/hw_train_s6k128_c6.hmm.seq', 'hw/hw_train_s6k128_c7.hmm.seq', 'hw/hw_train_s6k128_c8.hmm.seq'};
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
    disp('Done writing sequences...')

    clear('labels');
    testSeqFiles = {'hw/hw_test_s6k128_c1.hmm.seq', 'hw/hw_test_s6k128_c2.hmm.seq', 'hw/hw_test_s6k128_c3.hmm.seq', 'hw/hw_test_s6k128_c4.hmm.seq', 'hw/hw_test_s6k128_c5.hmm.seq', 'hw/hw_test_s6k128_c6.hmm.seq', 'hw/hw_test_s6k128_c7.hmm.seq', 'hw/hw_test_s6k128_c8.hmm.seq'};
    
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
    save(dumpFile);
else
    dumpFile = 'hw/hwHMM_s6k128.mat';
    load(dumpFile);
    disp(strcat('Done loading model from ', dumpFile));
end

MAKE_INIT = true;
initProbabs = {'hw/hw_train_s6k128_c1.hmm', 'hw/hw_train_s6k128_c2.hmm', 'hw/hw_train_s6k128_c3.hmm', 'hw/hw_train_s6k128_c4.hmm', 'hw/hw_train_s6k128_c5.hmm', 'hw/hw_train_s6k128_c6.hmm', 'hw/hw_train_s6k128_c7.hmm', 'hw/hw_train_s6k128_c8.hmm'};
if MAKE_INIT
    for i=1: length(initProbabs)
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
            if j == 1
                dlmwrite(initProbabs{i}, [0.9 prob], '-append', 'delimiter', ' ');
                dlmwrite(initProbabs{i}, [0.1 prob], '-append', 'delimiter', ' ');
            elseif j<numStates
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

DO_HMM_TRAINING = true;
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

modelFiles = {'hw/hw_train_s6k128_c1.hmm.seq.hmm', 'hw/hw_train_s6k128_c2.hmm.seq.hmm', 'hw/hw_train_s6k128_c3.hmm.seq.hmm', 'hw/hw_train_s6k128_c4.hmm.seq.hmm', 'hw/hw_train_s6k128_c5.hmm.seq.hmm', 'hw/hw_train_s6k128_c6.hmm.seq.hmm', 'hw/hw_train_s6k128_c7.hmm.seq.hmm', 'hw/hw_train_s6k128_c8.hmm.seq.hmm'};
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

printmat(C, 'Confution Matrix', 'ActCLASS1 CLASS2 CLASS3 CLASS4 CLASS5 CLASS6 CLASS7 CLASS8', 'PredCLASS1 CLASS2 CLASS3 CLASS4 CLASS5 CLASS6 CLASS7 CLASS8' );
Accuracy=(sum(diag(C)))/(sum(sum(C)))*100;
disp('ACCURACY(%)=');disp(Accuracy);

% k=numClasses;
% D=C;D(1:k+1:k*k) = 0;

% for i=1:k
%     Pclass(i)=C(i,i)/sum(C(i,:));
%     IError(i)=sum(D(i,:))/sum(C(i,:));
%     EError(i)=sum(D(:,i))/sum(C(:,i));
% end

% PE=horzcat(Pclass',IError',EError');
% printmat(PE, 'Precision Error', 'CLASS1 CLASS2 CLASS3 CLASS4 CLASS5 CLASS6 CLASS7 CLASS8', 'Precision inclusionEr exclusionEr' );
% scores = result(:, 1:end-2);
% for i=1:k
%     z(:,i)=((scores(:,i))-mean(scores(:,i)))/std(scores(:,i));
% end

% targetScores = 0;
% for i=1:k
%     targetScores = vertcat(targetScores, z(find(trueClass(:, 1) == i), i));
% end
% targetScores(1) = [];
% nonTargetScores = setdiff(reshape(z, [], 1), targetScores);
% figure;
% [f1, g1]=Compute_DET(targetScores, nonTargetScores);
% Plot_DET(f1,g1, 'r');
% set(get(gca,'XLabel'),'String','False positive rate');
% set(get(gca,'YLabel'),'String','Miss detection rate');
% title('DET curve for spiral data');

% %ROC Curve
% Tscore=vertcat(targetScores,nonTargetScores);
% NLabel1(1:length(targetScores))=1;
% NLabel2(1:length(nonTargetScores))=0;
% NLabel=vertcat(NLabel1',NLabel2');
% [X1,Y1] = perfcurve(NLabel,Tscore,1);
% figure;
% plot(X1,Y1);
% title('ROC spiral data');
% xlabel('False Positive rate');ylabel('True positive rate');

% <<======================== Performace metrics ends ============================>>
