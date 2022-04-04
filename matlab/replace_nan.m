function [out_arr] = replace_nan(arr)
% This fucntion replaces NaN values in the input array with the 
% valu of the nearest neighbor to each NaN value
% 

nan_locations = isnan(arr);
nan_idxs = find(nan_locations);
non_nan_idxs = setdiff(1:numel(arr), nan_idxs);

% Get the x,y of all other locations that are non nan.
[x, y] = ind2sub(size(arr), non_nan_idxs);

for index = 1 : length(nan_idxs)
    
  ii = nan_idxs(index);
    % Get the x,y location
  [x_nan,y_nan] = ind2sub(size(arr), ii);
  
  % Get distances of this location to all the other locations
   distances = sqrt((x_nan-x).^2 + (y_nan - y) .^ 2);
  [~, sortedIndexes] = sort(distances, 'ascend');
  % The closest non-nan value will be located at index sortedIndexes(1)
  indexOfClosest = sortedIndexes(1);
  
  % Get the arr value there.
  goodValue = arr(x(indexOfClosest), y(indexOfClosest));
  % Replace the bad nan value in u with the good value.
  arr(x_nan,y_nan) = goodValue;
  
end

out_arr = arr;
end

