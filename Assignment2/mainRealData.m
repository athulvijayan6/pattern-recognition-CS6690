% @Author: Athul Vijayan
% @Date:   2014-09-22 18:14:58
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-09-25 19:18:10

clear('all');
clc
delim = ' ';
warning off all;

C1 = importdata('class1_rw.txt', delim); C1(:,end+1) = ones(size(C1,1),1);
C2 = importdata('class2_rw.txt', delim); C2(:,end+1) = 2*ones(size(C2,1),1);
C3 = importdata('class3_rw.txt', delim); C3(:,end+1) = 3*ones(size(C3,1),1);

data = vertcat(C1, C2, C3);
data = data(randperm(size(data,1)),:);      % shuffle the data you take for training and testing
k = length(unique(data(:, end)));           % number of classes

trainData = zeros(1, size(data, 2));
crossValidationData = zeros(1, size(data, 2));
testData = zeros(1, size(data, 2));

for i=1:k
    numTraining = floor(size(find(data(:,end) == i), 1)*0.75);
    numCrossVal = floor(size(find(data(:,end) == i), 1)*0);
    numTest = size(find(data(:,end) == i), 1) - numTraining - numCrossVal;

    classIndices = find(data(:, end) == i);
    trainInd = classIndices(1:numTraining);
    crossValInd = classIndices(numTraining +1 : numTraining + numCrossVal);
    testInd = setdiff(classIndices, union(trainInd, crossValInd));

    trainData = vertcat(trainData, data(trainInd, :));
    crossValidationData = vertcat(trainData, data(crossValInd, :));
    testData = vertcat(trainData, data(testInd, :));

    [V{i}, D{i}] = eig(cov(data(classIndices, 1:end-1)));
end

testData = testData(2:end, :); crossValidationData = crossValidationData(2:end,:); 
testData = testData(2:end,:);


% % ================ Scatter plot ======================
% figure; hold on;
% plot(C1(:,1), C1(:,2), 'r^', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
% plot(C2(:,1), C2(:,2), 'go', 'MarkerSize', 4, 'MarkerFaceColor', 'g');
% plot(C3(:,1), C3(:,2), 'bd', 'MarkerSize', 4, 'MarkerFaceColor', 'b');

% set(get(gca,'XLabel'),'String','Feature 1');
% set(get(gca,'YLabel'),'String','Feature 2');
% title('Scatter plot of Real World Data');
% legend('Class 1', 'Class 2', 'Class 3');
% hold off;
% % =====================matlab axes length Scatterplot ends =================

% ====================== Box plot ===========================
% for i=1:k
%     figure;
%     classData = data(find(data(:,end) == i), :);
%     boxplot([classData(:,1) classData(:,2)], 'labels',{['Class ',num2str(i),' Feature 1'], ['Class ',num2str(i), 'Feature 2']});
%     title(['Box plot showing outliers in Class ',num2str(i),' data']);
% end
% ========================= Box plot ends===========================

% ========================= Outlier removal ======================
% figure; hold on;
colors = ['r', 'g', 'b'];
numout = 0;
for j=1:3
    
    classData = trainData(find(trainData(:,end) == j), 1:end-1);
    % figure(1); hold on
    % plot(classData(:,1), classData(:,2), '*', 'MarkerSize', 5, 'Color', colors(j));
    axes = axis();
    h = floor(0.25*size(classData, 1));
    sigma = cov(classData);
    mu = mean(classData);
    for l=1:h
        sigmadash = cov(classData(l:size(classData, 1)-h, :));
        if det(sigmadash) < det(sigma)
            sigma = sigmadash;
            mu = mean(classData(l:size(classData, 1)-h, :));
        end
    end    
    invsigma = inv(sigma);
    bound = inv(chi2pdf(0.975, 2));
    for i=1:size(classData, 1)
        x = classData(i, :);
        mahaDist = (x - mu)*invsigma*(x-mu)';        
        if mahaDist > bound
            classData(i,:) = mu;
            numout = numout + 1;
        end
    end
    trainData(find(trainData(:,end) == j), 1:end-1) = classData;
    % figure(2); hold on;
    % plot(classData(:,1), classData(:,2), '*', 'MarkerSize', 5, 'Color', colors(j)); 
    axis(axes);
end

for mods = 1:5;
    [model] = BuildBaysianModel(trainData, mods);

    x1 = linspace(min(testData(:, 1))-2, max(testData(:, 1))+2, 100);
    x2 = linspace(min(testData(:, 2))-2, max(testData(:, 2))+2, 100);

    [X1, X2] = meshgrid(x1, x2);

    idx = BayesianClassify(model, [X1(:) X2(:)]);
    idx = idx(:, 1);
    idx = reshape(idx, length(x2),length(x1));

    % ====================== Plot gaussian starts =============================
    figure; hold on;
    view(3);
    mu = model{1}{1}; Sigma = model{1}{2};
    F = mvnpdf([X1(:) X2(:)],mu,Sigma);
    F = reshape(F,length(x2),length(x1));
    surf(x1,x2,F, idx);

    mu = model{2}{1}; Sigma = model{2}{2};
    F = mvnpdf([X1(:) X2(:)],mu,Sigma);
    F = reshape(F,length(x2),length(x1));
    surf(x1,x2,F, idx);

    mu = model{3}{1}; Sigma = model{3}{2};
    F = mvnpdf([X1(:) X2(:)],mu,Sigma);
    F = reshape(F,length(x2),length(x1));
    surf(x1,x2,F, idx);

    set(get(gca,'XLabel'),'String','Feature 1');
    set(get(gca,'YLabel'),'String','Feature 2');
    set(get(gca,'ZLabel'),'String','scores');
    set(get(gca,'Title'),'String',['Gaussian for each class with decision boundary for case ', num2str(mods)]);
    annotation('textbox', [0.15 0.65 0.1 0.08], 'String', {'Red --> Class 1', 'Green --> Class 2', 'Blue --> Class 3'}, 'FitBoxToText','on');
    map = [1 0 0; 0 1 0; 0 0 1];
    colormap(map);

    % ====================== Plot gaussian ends =============================

    % <<====================== Plot contours ================================

    % figure; hold on;
    % axis equal;
    % imagesc(x1, x2, idx);
    % c1Plot = plot(testData(find(testData(:, end) == 1), 1), testData(find(testData(:, end) == 1), 2), 'rs');
    % set(c1Plot,'MarkerFaceColor','r'); set(c1Plot,'MarkerSize',3);
    % c2Plot = plot(testData(find(testData(:, end) == 2), 1), testData(find(testData(:, end) == 2), 2), 'go');
    % set(c2Plot,'MarkerFaceColor','g'); set(c2Plot,'MarkerSize',3);
    % c3Plot = plot(testData(find(testData(:, end) == 3), 1), testData(find(testData(:, end) == 3), 2), 'bd');
    % set(c3Plot,'MarkerFaceColor','b'); set(c3Plot,'MarkerSize',3);
    % legend('Class 1 Test data', 'Class 2 Test data', 'Class 3 Test data');

    % mu = model{1}{1}; Sigma = model{1}{2};
    % F = mvnpdf([X1(:) X2(:)],mu,Sigma);
    % F = reshape(F,length(x2),length(x1));
    % contour(x1,x2,F, 30);

    % mu = model{2}{1}; Sigma = model{2}{2};
    % F = mvnpdf([X1(:) X2(:)],mu,Sigma);
    % F = reshape(F,length(x2),length(x1));
    % contour(x1,x2,F, 30);

    % mu = model{3}{1}; Sigma = model{3}{2};
    % F = mvnpdf([X1(:) X2(:)],mu,Sigma);
    % F = reshape(F,length(x2),length(x1));
    % contour(x1,x2,F, 30);

    % set(get(gca,'XLabel'),'String','Feature 1');
    % set(get(gca,'YLabel'),'String','Feature 2');
    % set(get(gca,'Title'),'String',['Contours and test data for each class with decision boundary for Case ', num2str(mods)]);
    % annotation('textbox', [0.15 0.65 0.18 0.08], 'String', 'Class 1 Region', 'FitBoxToText','on');
    % annotation('textbox', [0.35 0.65 0.18 0.08], 'String', 'Class 2 Region', 'FitBoxToText','on');
    % annotation('textbox', [0.55 0.65 0.18 0.08], 'String', 'Class 3 Region', 'FitBoxToText','on');

    % for m=1:k
    %     classIndices = find(data(:, end) == m);
    %     classData = data(classIndices, 1:2);
    %     mu = mean(classData);
    %     [v,d] = eig(classData'*classData);
    %     plot([mu(1) mu(1) + 60*v(1,1)],[mu(2) mu(2) + 60*v(2,1)],'w-','LineWidth',2);   % first eigenvector
    %     plot([mu(1) mu(1) + 60*v(1,2)],[mu(2) mu(2) + 60*v(2,2)],'y-' ,'LineWidth',2);   % first eigenvector
    % end
    % annotation('textbox', [0.01 0.1 0.18 0.08], 'String', 'The lines at the middle of each distribution shows the eigen vectors of each class', 'FitBoxToText','on');
    % hold off

    % % <<=========================== contour ends =========================

    % % <<=============================Performance eval start ===============

    % classLabels = BayesianClassify(model, testData(:,1:end-1));
    % k = length(unique(testData(:, end))); % number of classes

    % trueClass=testData(:,end);
    % predClass=classLabels(:,1);

    % [C,or]= confusionmat(trueClass, predClass);

    % printmat(C, 'Confution Matrix', 'ActCLASS1 CLASS2 CLASS3', 'PredCLASS1 CLASS2 CLASS3' );
    % Accuracy=(sum(diag(C)))/(sum(sum(C)))*100;
    % disp('ACCURACY(%)=');disp(Accuracy);

    % D=C;D(1:k+1:k*k) = 0;
    % for i=1:k
    %     Pclass(i)=C(i,i)/sum(C(i,:));
    %     IError(i)=sum(D(i,:))/sum(C(i,:));
    %     EError(i)=sum(D(:,i))/sum(C(:,i));
    % end

    % PE=horzcat(Pclass',IError',EError');
    % printmat(PE, 'Precision Error', 'CLASS1 CLASS2 CLASS3', 'Precision inclusionEr exclusionEr' );

    % for i=1:k
    %     z(:,i) = ((classLabels(:,i+1))-mean(classLabels(:,i+1)))/std(classLabels(:,i+1));
    % end
    % colors = ['r', 'g', 'b', 'm', 'c'];

    % % ============================== DET and ROC for each class =======================
    % for i=1:k
    %     figure(i);hold on;
    %     [X(:,i),Y(:,i)] = perfcurve(trueClass, z(:,i),i);
    %     p = plot(X(:,i),Y(:,i));
    %     set(p, 'Color', colors(mods)); 
    %     title(['ROC Curves for Class ', num2str(i)]);
    %     xlabel('False Positive rate');ylabel('True positive rate');
    %     legend('Algo 1','Algo 2','Algo 3', 'Algo 4','Algo 5','Location','NorthEastOutside');
    % end
 
    % for i=1:k
    %     figure(k+i); hold on;
    %     targetScores = z(find(classLabels(:, 1) == i), i);
    %     nonTargetScores = setdiff(z(:, i), targetScores);
    %     [f1, g1]=Compute_DET(targetScores, nonTargetScores);
    %     Plot_DET(f1,g1, colors(mods));
    %     set(get(gca,'XLabel'),'String','False Positive Rate');
    %     set(get(gca,'YLabel'),'String','Missed detection rate');
    %     title(['Showing DET curves for Class ', num2str(i)]);
    %     legend('Algo 1','Algo 2','Algo 3', 'Algo 4','Algo 5','Location','NorthEastOutside');
    % end
    % % ============================== DET and ROC for each class ends =======================

    % ======================DET and ROC for each algorithm =============
    % figure(mods);hold on;
    % for i=1:k
    %     [X(:,i),Y(:,i)] = perfcurve(trueClass, z(:,i),i);
    %     p = plot(X(:,i),Y(:,i));
    %     set(p, 'Color', colors(i)); 
    % end
    % title(['ROC Curves for Algorithm ', num2str(mods)]);
    % xlabel('False Positive rate');ylabel('True positive rate');
    % legend('Class 1', 'Class 2', 'Class 3','Location','NorthEastOutside');
    % hold off
 
    % figure(5+mods); hold on;
    % for i=1:k
    %     targetScores = z(find(classLabels(:, 1) == i), i);
    %     nonTargetScores = setdiff(z(:, i), targetScores);
    %     [f1, g1]=Compute_DET(targetScores, nonTargetScores);
    %     Plot_DET(f1,g1, colors(i));
        
    % end
    % set(get(gca,'XLabel'),'String','False Positive Rate');
    % set(get(gca,'YLabel'),'String','Missed detection rate');
    % title(['Showing DET curves for Algorithm ', num2str(mods)]);
    % legend('Class 1', 'Class 2', 'Class 3','Location','NorthEastOutside');
    % hold off
    % ======================DET and ROC for each algorithm ends =============

    % <<=============================Performance eval ends ===============
end