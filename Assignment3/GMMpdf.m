% @Author: Athul Vijayan
% @Date:   2014-10-18 06:45:20
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-27 17:21:35

%% GMMpdf: The function extract parameters for a Gaussian Mixture Model using unlabelled training data.
% Function usage
% Inputs
    % data --------> training data
    % k -----------> Number of Gaussians in the mixture
    % (optional) cov_type ------> nature of covariance. either 'full' or 'diagonal'
    % (optional) max_iter ----> Maximum iteration in case of very slow convergence (default = 100)
    % (optional) delta_cutoff-> Tell wen to stop iterations (default = 1e-5)
% Outputs
    % muHat --------> MLE estimate of mean of each gaussian as kx1 cell
    % sigmaHat --------> MLE estimate of covariance of each gaussian as kx1 cell
function [muHat, sigmaHat, piHat, likelihood] = GMMpdf(data, k, cov_type, max_iter, delta_cutoff)

    if nargin ==2
        cov_type = 'diagonal';
        max_iter = 100;
        delta_cutoff = 1e-5;
    elseif nargin == 3
        max_iter = 100;
        delta_cutoff = 1e-5;
    elseif nargin == 4
        delta_cutoff = 1e-5;
    end
    [pts, dim] = size(data);
    % First Apply kmeans to find initial parameters
    [ label, centroids, ~] = mykmeans( data, k, 10);
    
    for i=1:k                       %initial parameters from k means
        muHat{i} = centroids(i, :);
        sigmaHat{i} = diag(var(data(find(label==i), :)));
        piHat{i} = size(find(label==i), 2)/ size(data, 1);
    end

    iter_no = 1;    
    delta = inf;
    l_old = inf;
    while ((iter_no<max_iter) && (delta > delta_cutoff))
        % Calculate the latent variable from old parameters  (E step in EM)
        % tStart = tic;
        for i=1:pts
            for j=1:k
                gamma(i, j) = piHat{j}*mvnpdf(data(i, :), muHat{j}, sigmaHat{j});
            end
            gamma(i, :) = gamma(i, :)./sum(gamma(i, :));
        end
        % tElapsed = toc(tStart)
        % Reestimate parameters using latent variable (M step in EM)
        % tStart = tic;

        for j=1:k
            N(j) = sum(gamma(:, j));

            muHat{j} = 0;
            for i=1:pts
                muHat{j} = muHat{j} + gamma(i, j)* data(i, :);
            end
            muHat{j} = muHat{j}./N(j);

            sigmaHat{j} = 0;            
            if strcmp(cov_type, 'full')
                for i=1:pts
                    sigmaHat{j} = sigmaHat{j} + gamma(i, j)* (data(i, :) - muHat{j})'*(data(i, :) - muHat{j});
                end
                sigmaHat{j} = sigmaHat{j}/N(j);
            elseif strcmp(cov_type, 'diagonal')
                for i=1:pts
                    sigmaHat{j} = sigmaHat{j} + gamma(i, j)* diag((data(i, :) - muHat{j}).^2);
                end
                sigmaHat{j} = sigmaHat{j}./ N(j);
            end
        end
        % tElapsed=toc(tStart)
        % find new likelihood for convergence test
        % tStart= tic;
        l_new = 0;
        for i=1:pts
            g = 0;
            for j=1:k
                g = g + piHat{j}*mvnpdf(data(i, :), muHat{j}, sigmaHat{j});
            end
            l_new = l_new + log(g);
        end
        % tElapsed = toc(tStart)
        delta = abs(l_new - l_old);
        likelihood(iter_no) = l_new;
        l_old = l_new;
        iter_no = iter_no +1;
    end
end
