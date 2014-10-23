% @Author: Athul Vijayan
% @Date:   2014-10-21 19:28:55
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-22 17:24:33


clear('all');
clc
DO_TRAINING = false;
dumpFile = 'spiralGMM_n5.mat';

if DO_TRAINING
    filename = 'spiralData.txt';
    delim = ' ';

    numNorms = [10 10];
    data = importdata(filename, delim);
    data(find(data(:, end) == 1), end) = 2;
    data(find(data(:, end) == 0), end) = 1;
    data = data(randperm(size(data,1)),:);
    trainData = zeros(1, 3);
    testData = zeros(1, 3);
    for i=1:2
        classIndices = find(data(:, end) == i);
        trainInd = classIndices(1:0.75*length(classIndices));
        testInd = setdiff(classIndices, trainInd);

        trainData = vertcat(trainData, data(trainInd, :));
        testData = vertcat(testData, data(testInd, :));
    end
    trainData(1, :) = []; testData(1, :) = []; 

    [model, likelihood] = GMMtrain(trainData, numNorms, 'diagonal', 50, 1e-3);
    clearvars -except model testData trainData data dumpFile likelihood numNorms s;
    save(dumpFile);
else
    load(dumpFile)
end


plot(data(find(data(:, end) == 1), 1), data(find(data(:, end) == 1), 2), 'r*');
hold on
plot(data(find(data(:, end) == 2), 1), data(find(data(:, end) == 2), 2), 'gd');

[classLabels, scores] = GMMclassify(model, testData(:, 1:2));
nnz(classLabels' - testData(:, end))
c = ['r' 'g'];

% ================== Plot data========================
% for i=1:length(testData)
%     plot(testData(i, 1), testData(i, 2), '*', 'Color', c(classLabels(i)));
% end

% ================== Plot data ends========================


% ===================Contour plot starts ========================

% x1 = linspace(min(testData(:, 1)), max(testData(:, 1)), 100);
% x2 = linspace(min(testData(:, 2)), max(testData(:, 2)), 100);

% [X1, X2] = meshgrid(x1, x2);
% [idx, ~] = GMMclassify(model, [X1(:) X2(:)]);
% idx = idx(1,:);
% idx = reshape(idx, length(x2),length(x1));

% imagesc(x1, x2, idx);
% for i=1:size(model, 1)
%     for j=1:length(model{i}{1})
%         F = mvnpdf([X1(:) X2(:)],model{i}{1}{j},model{i}{2}{j});
%         F = reshape(F,length(x2),length(x1));
%         c = ['r' 'g'];
%         contour(x1,x2,F, 2, 'Color', c(i));
%         c = ['c' 'b'];
%         plot(model{i}{1}{j}(1), model{i}{1}{j}(2),  'co', 'MarkerSize', 8, 'MarkerFaceColor', c(i));
%         set(get(gca,'XLabel'),'String','Feature 1');
%         set(get(gca,'YLabel'),'String','Feature 2');
%         title('Contours of the Gaussians with diagonal covariance fitted with GMM');
%         legend('Class 1', 'Class 2');
%     end
% end
% hold off
% ===========================contor plot ends====================

% <<======================== Performace Curves starts ============================>>
trueClass=testData(:,end);
predClass=classLabels';

[C,or]= confusionmat(trueClass, predClass);

printmat(C, 'Confution Matrix', 'ActCLASS1 CLASS2 CLASS3', 'PredCLASS1 CLASS2 CLASS3' );
Accuracy=(sum(diag(C)))/(sum(sum(C)))*100;
disp('ACCURACY(%)=');disp(Accuracy);

k=2;
D=C;D(1:k+1:k*k) = 0;

for i=1:k
    Pclass(i)=C(i,i)/sum(C(i,:));
    IError(i)=sum(D(i,:))/sum(C(i,:));
    EError(i)=sum(D(:,i))/sum(C(:,i));
end

PE=horzcat(Pclass',IError',EError');
printmat(PE, 'Precision Error', 'CLASS1 CLASS2 CLASS3', 'Precision inclusionEr exclusionEr' );
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
