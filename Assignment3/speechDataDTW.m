% @Author: Athul Vijayan
% @Date:   2014-10-23 12:00:41
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-23 12:28:12

paths = {'digit_data/four/' ; 'digit_data/seven/'; 'digit_data/nine/'; 'digit_data/two/'; 'digit_data/five/'};
for i=1: size(paths, 1)
    allFiles = struct2table(dir(strcat(paths{i}, '*.txt')));
    allFiles = table2array(allFiles(:, 1));
    allFiles = allFiles(randperm(size(allFiles, 1)), :);
    trainFiles{i} = allFiles(1:floor(0.2*length(allFiles)), :);
    testFiles{i} = setdiff(allFiles, trainFiles{i});
end

for i=1:length(trainFiles)
    template{i} = importdata(strcat(paths{i}, trainFiles{i}{randi(length(trainFiles{i}))}), ' ');
end

result = zeros(1, length(testFiles) +2);
for i=1:length(testFiles)
    for j=1:length(testFiles{i})
        a = importdata(strcat(paths{i}, testFiles{i}{j}), ' ');
        for m =1:length(template)
            [scores(m), ~] = myDTW(a, template{m});
        end
        scores = scores./sum(scores);
        predClass = find(scores == min(scores));
        result = vertcat(result, [scores predClass i]);
    end
end
result(1,:) = [];

