% @Author: Athul Vijayan
% @Date:   2014-09-15 17:49:21
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-09-22 00:35:24
clear all
clc

% filename = 'data1a.txt';
delim = ' ';
numTraining = 400;
numCrossVal = 0;
numTest = 500 - numTraining - numCrossVal;

C1 = importdata('class1_nl.txt', ' ');
C2 = importdata('class2_nl.txt', ' ');
C3 = importdata('class3_nl.txt', ' ');

% data = importdata(filename, delim);
data = vertcat(C1, C2, C3);
data(:, size(data, 2) + 1) = zeros(size(data,1), 1);
for n=1:size(data, 1)
    data(n, end) = floor(n/500) + 1;
    if rem(n, 500) == 0
        data(n, end) = data(n, end) - 1;
    end
end
k = length(unique(data(:, end))); % number of classes
V = cell(k, 1); D = cell(k, 1);
for i=min(data(:,end)): max(data(:, end))
    classIndices = find(data(:, end) == i);
    trainInd = classIndices(1:numTraining);
    crossValInd = classIndices(numTraining +1 : numTraining + numCrossVal);
    testInd = setdiff(classIndices, union(trainInd, crossValInd));

    trainData((i-1)*numTraining + 1: i*numTraining, :) = data(trainInd, :);
    crossValidationData((i-1)*numCrossVal + 1: i*numCrossVal, :) = data(crossValInd, :);
    testData((i-1)*numTest + 1: i*numTest, :) = data(testInd, :);

    [V{i}, D{i}] = eig(cov(data(classIndices, 1:end-1)));
end


for mods=1:5
[model] = BuildBaysianModel(trainData, mods);

classHat = BayesianClassify(model, testData(:,1:end-1));
nnz(classHat - testData(:,end))


x1 = linspace(min(testData(:, 1))-2, max(testData(:, 1))+2, 200);
x2 = linspace(min(testData(:, 2))-2, max(testData(:, 2))+2, 200);

[X1, X2] = meshgrid(x1, x2);

idx = BayesianClassify(model, [X1(:) X2(:)]);
idx = reshape(idx, length(x2),length(x1));

% % << ============================ contour plot starts ===================>

figure; hold on;
axis equal;
imagesc(x1, x2, idx);
c1Plot = plot(testData(find(testData(:, end) == 1), 1), testData(find(testData(:, end) == 1), 2), 'rs');
set(c1Plot,'MarkerFaceColor','r'); set(c1Plot,'MarkerSize',8);
c2Plot = plot(testData(find(testData(:, end) == 2), 1), testData(find(testData(:, end) == 2), 2), 'go');
set(c2Plot,'MarkerFaceColor','g'); set(c2Plot,'MarkerSize',8);
c3Plot = plot(testData(find(testData(:, end) == 3), 1), testData(find(testData(:, end) == 3), 2), 'bd');
set(c3Plot,'MarkerFaceColor','b'); set(c3Plot,'MarkerSize',8);
legend('Class 1 Test data', 'Class 2 Test data', 'Class 3 Test data');

mu = model{1}{1}; Sigma = model{1}{2};
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
contour(x1,x2,F, 30);

mu = model{2}{1}; Sigma = model{2}{2};
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
contour(x1,x2,F, 30);

mu = model{3}{1}; Sigma = model{3}{2};
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
contour(x1,x2,F, 30);

set(get(gca,'XLabel'),'String','Feature 1');
set(get(gca,'YLabel'),'String','Feature 2');
set(get(gca,'Title'),'String',['Contours and test data for each class with decision boundary for Case ', num2str(mods)]);
annotation('textbox', [0.15 0.65 0.18 0.08], 'String', 'Class 1 Region', 'FitBoxToText','on');
annotation('textbox', [0.35 0.65 0.18 0.08], 'String', 'Class 2 Region', 'FitBoxToText','on');
annotation('textbox', [0.55 0.65 0.18 0.08], 'String', 'Class 3 Region', 'FitBoxToText','on');
for m=1:k
    classIndices = find(data(:, end) == m);
    classData = data(classIndices, 1:2);
    mu = mean(classData)
    % classData = classData - repmat(mu,length(classData),1);
    [v,d] = eig(classData'*classData);
    % if (d(1,1) < d(2,2))
    %     tmp = v(:,1);
    %     v(:,1) = v(:, 2);
    %     v(:,2) = tmp;
    % end
    plot([mu(1) mu(1) + 2*v(1,1)],[mu(2) mu(2) + 2*v(2,1)],'w-','LineWidth',3);   % first eigenvector
    plot([mu(1) mu(1) + 2*v(1,2)],[mu(2) mu(2) + 2*v(2,2)],'y-' ,'LineWidth',3);   % first eigenvector
end
annotation('textbox', [0.01 0.1 0.18 0.08], 'String', 'The lines at the middle of each distribution shows the eigen vectors of each class', 'FitBoxToText','on');

hold off

%  % << ============================ contour plot ends ===================>


% << ======================= Gaussian plots starts============================>>
% figure; hold on;
% view(3);
% mu = model{1}{1}; Sigma = model{1}{2};
% F = mvnpdf([X1(:) X2(:)],mu,Sigma);
% F = reshape(F,length(x2),length(x1));
% surf(x1,x2,F, idx);

% mu = model{2}{1}; Sigma = model{2}{2};
% F = mvnpdf([X1(:) X2(:)],mu,Sigma);
% F = reshape(F,length(x2),length(x1));
% surf(x1,x2,F, idx);

% mu = model{3}{1}; Sigma = model{3}{2};
% F = mvnpdf([X1(:) X2(:)],mu,Sigma);
% F = reshape(F,length(x2),length(x1));
% surf(x1,x2,F, idx);

% set(get(gca,'XLabel'),'String','Feature 1');
% set(get(gca,'YLabel'),'String','Feature 2');
% set(get(gca,'ZLabel'),'String','scores');
% set(get(gca,'Title'),'String',['Gaussian for each class with decision boundary for case ', num2str(mods)]);
% annotation('textbox', [0.15 0.65 0.1 0.08], 'String', {'Red --> Class 1', 'Green --> Class 2', 'Blue --> Class 3'}, 'FitBoxToText','on');
% % saveas(gca, ['data2gaussian',num2str(mods),'.eps'],'epsc');
% map = [1 0 0; 0 1 0; 0 0 1];
% colormap(map);

% << ======================= Gaussian plots ends ============================>>
end