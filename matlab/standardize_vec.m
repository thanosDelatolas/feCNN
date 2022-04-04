function [standardized_vector] = standardize_vec(in_vector)
% This function standardize the input vector

standardized_vector = (in_vector - mean(in_vector))/ std(in_vector);
end

