function [normilized_vector] = normalize_vec(in_vector)
% This function normilize the input vector to range 0-1

normilized_vector = (in_vector - min(in_vector))/(max(in_vector) - min(in_vector));
end

