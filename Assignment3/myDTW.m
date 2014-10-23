% @Author: Athul Vijayan
% @Date:   2014-10-22 23:45:42
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-10-23 12:53:19

%% myDTW: function description
function [d, costMat] = myDTW(test, ref, max_steps)
    if nargin < 3
        max_steps = Inf;
    end
    
    n = length(test);
    m = length(ref);
    max_steps = max(max_steps, abs(n-m));

    costMat = zeros(n + 1, m + 1) + Inf;
    costMat(1, 1) = 0;
    for i=1:n
        if (max_steps == Inf)
            for j=1:m
                costMat(i +1,  j+1) = min([costMat(i, j+1), costMat(i+1, j), costMat(i, j) ]) + norm(test(i, :) - ref(j, :));
            end
        else
            for j= max(i- max_steps, 1): min(i + max_steps, m)
                costMat(i +1,  j+1) = min([costMat(i, j+1), costMat(i+1, j), costMat(i, j) ]) + norm(test(i, :) - ref(j, :));
            end
        end
    end
    d = costMat(n+1, m+1);
end


