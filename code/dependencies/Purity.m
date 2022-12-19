%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purity refer to https://en.wikipedia.org/wiki/Cluster_analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pur = Purity(y_true, y_pred)
    if size(y_true) ~= size(y_pred)
        error('The input size is not same!');
    end
    num = length(y_true);
    labels = unique(y_true);
    [~,labels_size] = size(labels);

    % the number of classes with the largest number of true classifications in each predicted class
    max_sum = zeros(1,labels_size); 

    for i = 1:labels_size
        % The position of each class in the true and predicted arrays
        [~,idx] = find(y_pred == labels(i)); 
        max_number = mode(y_true(idx));
        max_sum(i) = length(find(y_true(idx) == max_number));
    end
    pur = sum(max_sum)/num;
return
