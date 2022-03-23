function [locations] = find_multiple_soucres(source, cd_matrix)
% This function finds the seed dipoles when multiple sources operate simultaneously

   % find local maximas
    [pks,locs] = findpeaks(source);
    locations = [];
    mean_source = mean(source);
    activations = 0;
    % distance from dipole in mm
    activation_radius = 55;
    for ii=1:length(pks)
        % ignore dipoles with small amplitude
        if pks(ii) < mean_source
            continue
        end
        cont =0;
        % ignore dipoles in a neighborhood
        for kk=1:length(locations)
            if distance_3d_space(cd_matrix(locs(ii),1:3),cd_matrix(locations(kk),1:3)) <= activation_radius
                cont=1;
                break;
             end
        end
        if cont == 1
            continue
        end
        
        dipole = locs(ii);
        neighborhood_idxs = [];
        dipoles_in = 0; % dipoles in neighborhood
        % get the neighborhood of the dipole
        for jj=1:length(source)
            if distance_3d_space(cd_matrix(jj,1:3),cd_matrix(dipole,1:3)) <= activation_radius
                dipoles_in = dipoles_in +1;
                neighborhood_idxs(dipoles_in) = jj;
            end
        end
        
        
        neighborhood_vals = source(neighborhood_idxs);
        [~,idx_max] = max(neighborhood_vals);
        seed_dipole = neighborhood_idxs(idx_max);
        
        activations=activations+1;
        locations(activations) = seed_dipole;
    end
     
    % get the two dipoles with the maximum amplitude
    if length(locations) > 2
        [~,idx_max] = maxk(source(locations),2);
        locations = locations(idx_max);
    end
    

end