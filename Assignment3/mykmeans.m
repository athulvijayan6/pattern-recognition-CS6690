% @Author: Athul Vijayan
% @Date:   2014-10-16 13:44:18
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-18 07:41:40

function [ label, centroids, D] = mykmeans( data, k , max_iter, delta_cutoff)

% the function classifies points given as argument to clusters and assign labels to each point
% Label is the cluster number to which a point belongs

% function takes arguments
    % data ------> data matrix. columns are dimensions and rows are points in the dimensional space
    % k -----> number of clusters
    % max_iter (optional)----------> maximum iterations
% and returns
    % label  ------> the cluster to which each data point labelled to
    % centroids -----> centroid of final clusters

if nargin ==2
    max_iter = 100;
    delta_cutoff = 1e-5;
elseif nargin == 3
    delta_cutoff =1e-5;
end

[points, dim] = size(data);

% for initial points we use random points from data
seq = 1:points;
for i=1:k
    randpt = randi(numel(seq));
    centroids(i,:) = data(seq(randpt),:);   % take 4 random points
    seq(randpt) = [];
end

iter_no = 1;
delta = inf;
lastCentroids = centroids;

while ((iter_no<max_iter) && (delta > delta_cutoff))
    D(iter_no) = 0;
    for i=1:points    % iterate through points in data for calculating labels
        for j=1:k
            dist(j) = norm(data(i,:) - centroids(j,:)); 
        end
        [val, label(i)] = min(dist);   % label will have the centroid to which every point is included.
        D(iter_no) = D(iter_no) + val;
    end
    for j=1:k
        centroids(j, :) = mean(data(find(label==j),:));   %find new centroids
    end
    delta = norm(lastCentroids - centroids)^2;
    lastCentroids = centroids;
    iter_no = iter_no + 1; % keep count of iterations
end




