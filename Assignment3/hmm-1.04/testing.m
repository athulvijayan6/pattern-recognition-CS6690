clear all
clc

filename = 'data1c.txt';
delim = ' ';
numTraining = 350;
numTest = 500 - numTraining;

% C1 = importdata('class1_nl.txt', ' ');
% C2 = importdata('class2_nl.txt', ' ');
% C3 = importdata('class3_nl.txt', ' ');

data = importdata(filename, delim);
% data = vertcat(C1, C2, C3);
data(:, size(data, 2) + 1) = zeros(size(data,1), 1);
for n=1:size(data, 1)
    data(n, end) = floor(n/500) + 1;
    if rem(n, 500) == 0
        data(n, end) = data(n, end) - 1;
    end
end

for i=min(data(:,end)): max(data(:, end))

    classIndices = find(data(:, end) == i);
    trainInd = classIndices(1:numTraining);
    testInd = setdiff(classIndices,trainInd);

    trainData((i-1)*numTraining + 1: i*numTraining, :) = data(trainInd, :);
    testData((i-1)*numTest + 1: i*numTest, :) = data(testInd, :);
end

[model1] = BuildBaysianModel(trainData, 1);
%[model2] = BuildBaysianModel(trainData, 2);
%[model3] = BuildBaysianModel(trainData, 3);
%[model4] = BuildBaysianModel(trainData, 4);
%[model5] = BuildBaysianModel(trainData, 5);

classLabels = BayesianClassify(model1, testData(:,1:end-1));
miss=nnz(classLabels(:,1) - testData(:,end));
x1=testData(:,end);
x2=classLabels(:,1);
k = length(unique(testData(:, end))); % number of classes
[C,or]= confusionmat(x1,x2);
printmat(C, 'Confution Matrix', 'ActCLASS1 CLASS2 CLASS3', 'PredCLASS1 CLASS2 CLASS3' );
Accuracy=(sum(diag(C)))/(sum(sum(C)))*100;
disp('ACCURACY(%)=');disp(Accuracy);
D=C;D(1:k+1:k*k) = 0;
for i=1:k
Pclass(i)=C(i,i)/sum(C(i,:));
IError(i)=sum(D(i,:))/sum(C(i,:));
EError(i)=sum(D(:,i))/sum(C(:,i));
end
PE=horzcat(Pclass',IError',EError');
printmat(PE, 'Precision Error', 'CLASS1 CLASS2 CLASS3', 'Precision inclusionEr exclusionEr' );
for i=1:k
z(:,i)=((classLabels(:,i+1))-mean(classLabels(:,i+1)))/std(classLabels(:,i+1));
end
figure(1);hold on;
colors = ['r', 'g', 'b', 'm'];
for i=1:k
    [X(:,i),Y(:,i)] = perfcurve(x1,z(:,i),i);
 p1 = plot(X(:,i),Y(:,i));
 set(p1, 'Color', colors(i));
end
title('ROC Curves');
xlabel('False Positive rate');ylabel('True positive rate');
legend('Class 1','Class 2','Class 3',...
'Location','NorthEastOutside');hold off;
figure(2);
[f1,g1]=Compute_DET(z(1:numTest,1),z(2*numTest+1:end,1));
Plot_DET(f1,g1,'r');hold on;
[f2,g2]=Compute_DET(z(numTest+1:2*numTest,2),vertcat(z(1:numTest,2),z(2*numTest+1:3*numTest,2)));
Plot_DET(f2,g2,'g');
[f3,g3]=Compute_DET(z(2*numTest+1:end,3),z(1:2*numTest,3));
Plot_DET(f3,g3,'b');
%grp2idx(x1,x2);
%[tp,fp,thres]=roc(x1,x2); 
%m2Clas = BayesianClassify(model2, testData(:,1:end-1));
%nnz(m2Clas - testData(:,end))
%m3Clas = BayesianClassify(model3, testData(:,1:end-1));
%nnz(m3Clas - testData(:,end))
%m4Clas = BayesianClassify(model4, testData(:,1:end-1));
%nnz(m4Clas - testData(:,end))
%m5Clas = BayesianClassify(model5, testData(:,1:end-1));
%nnz(m5Clas - testData(:,end))

%scores=classLabels(:,2);
%Error=sum(x1~=x2)/length(x2);
%for i=1:length(classLabel);
   % if x1=x2
%u=mean(scores);
%sig=std(scores);
%zz=(scores-u)/sig;
%z=zscore(scores);
%for i=1:k
%str1=['The precision of class' num2str(i)];    
%disp(str1);
%Pclass(i)=C(i,i)/sum(C(i,:));
%disp(Pclass(i));
%str2=['The inclusion error of class' num2str(i)];    
%disp(str2);
%IError(i)=sum(D(i,:))/sum(C(i,:));
%disp(IError(i));
%str3=['The exclusion error of class' num2str(i)];    
%disp(str3);
%EError(i)=sum(D(:,i))/sum(C(:,i));
%disp(EError(i));
%end







