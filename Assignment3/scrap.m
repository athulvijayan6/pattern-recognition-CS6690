% @Author: Athul Vijayan
% @Date:   2014-11-05 16:08:52
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-11-08 09:03:32


load('hwDataGMM_n24.mat')


numC = 2;

result = zeros(1, length(files) + 2);
for i=numC
    classData = testData(find(testData(:, end-1) == i), :);
    for j=1:max(classData(:,end))
        d = classData(find(classData(:, end) == j), 1:end-2);
        [~, s] = GMMclassify(model, d);
        scores = prod(s);
        scores = scores./sum(scores);
        predLabel = find(scores == max(scores));
        result = vertcat(result, [scores predLabel i]);
    end
end
result(1, :) = [];

c1 = testData(find(testData(:, end-1) == numC), :);
figure; hold on;
for i=1:max(c1(:, end))
    p = c1(find(c1(:, end) == i), 1:2);
    plot(p(:, 1), p(:, 2), 'r*');
end



x1 = linspace(min(p(:, 1)), max(p(:, 1)), 100);
x2 = linspace(min(p(:, 2)), max(p(:, 2)), 100);

[X1, X2] = meshgrid(x1, x2);

i = numC;

for j=1:length(model{i}{1})
    F = mvnpdf([X1(:) X2(:)],model{i}{1}{j},model{i}{2}{j});
    F = reshape(F,length(x2),length(x1));
    contour(x1,x2,F, 2, 'Color', 'c');
    plot(model{i}{1}{j}(1), model{i}{1}{j}(2),  'co', 'MarkerSize', 8, 'MarkerFaceColor', 'c');
    set(get(gca,'XLabel'),'String','normlised x');
    set(get(gca,'YLabel'),'String','normalised y');
    title('Contours of the Gaussians with diagonal covariance fitted - GMM with 12 gaussians');
end
