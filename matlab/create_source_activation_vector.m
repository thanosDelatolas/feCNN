function [source_activation,dipole_location] = create_source_activation_vector(data,method,cd_matrix)
% Normilizes and creates the vector for the source activation.
% data: the output of a source localization method.
% method: method used for source localization e.g. mne, sLORETA
% cd_matrix: the source sapce

if strcmp(method,'sLORETA')
    % find the average 3d-coordinates of the 100 dipoles with max amplityde
    [max_100_values, max_100_indexes] = maxk(data,100);

    coordinates = cd_matrix(max_100_indexes,1:3);
    average_coordinate = mean(coordinates,1);
    dipole_location = average_coordinate;
    dipole_value = mean(max_100_values);
    
    % find the closest dipole to the estimation
    activation_idx = -1;
    dist = Inf;
    x_target = dipole_location(1);
    y_target = dipole_location(2);
    z_target = dipole_location(3);
    for ii=1:length(cd_matrix)
        coord = cd_matrix(ii,1:3);
        xi = coord(1); yi = coord(2); zi = coord(3);
        distance_to_target = sqrt((xi-x_target)^2 + (yi - y_target)^ 2 + (zi - z_target)^ 2);

        if distance_to_target < dist
            dist = distance_to_target;
            activation_idx = ii;
        end

    end
    
else 
    [dipole_value, activation_idx] = max(data);
end

dipole_location = cd_matrix(activation_idx,1:3);
%cnn that predicts the electrical current
if strcmp(method,'nn')
    source_activation = data;
    
elseif strcmp(method,'loc_cnn') % new cnn that predicts the x,y,z coordinates
    min_dist = inf;
    dipole = -1;
    for ii=1:size(cd_matrix,1)
        if distance_3d_space(data,cd_matrix(ii,1:3)) < min_dist
            min_dist = distance_3d_space(data,cd_matrix(ii,1:3));
            dipole = ii;
        end
    end

    source_activation = -1 * ones(size(cd_matrix,1),1);
    source_activation(dipole) = 1;
else
    source_activation = (-1) * ones(size(data));
    source_activation(activation_idx) = dipole_value;
end
    
source_activation = normalize_vec(source_activation);




end

