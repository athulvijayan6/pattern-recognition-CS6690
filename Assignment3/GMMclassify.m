% @Author: Athul Vijayan
% @Date:   2014-10-18 18:00:35
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-23 10:27:56

%% GMMclassify: function description
function [classLabels, scores] = GMMclassify(model, testData)

    m = size(testData, 1); % number of examples
    n = size(testData, 2); % number of feature dimension
    k = size(model, 1); % number of classes

    for i=1:m
        x = testData(i, :);
        for j=1:k
            muHat = model{j}{1};
            sigmaHat = model{j}{2};
            piHat = model{j}{3};
            numNorms = length(muHat);
            P(j) = 0;
            for m=1:numNorms
                P(j) = P(j) + piHat{m}*mvnpdf(x, muHat{m}, sigmaHat{m});
            end
            P(j) = log(P(j));
        end
        scores(i, :) = P./sum(P);
        classLabels(i) = find(scores(i, :) == max(scores(i, :)));
    end
end
