clear; close all; clc;

import_fieldtrip();

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

eeg = double(readNPY('/media/thanos/Elements/thanos/sim_data/sim_type_1/downsampled_dipoles-25k/1e-15/eeg_4TeD.npy'));

%sources = double(readNPY('../../../Downloads/sources.npy'));

layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';

[sensors_1010, lay] = compatible_elec(sensor_labels, layout);

n_samples = size(eeg,2);


eeg_topos = zeros(67,67, n_samples);
eeg_Xi = zeros(67,67, n_samples);
eeg_Yi = zeros(67,67, n_samples);

w_bar = waitbar(0, 'Creating CNN input...');

tic;
for ii=1:n_samples
    eeg_s = (eeg(:,ii) - mean(eeg(:,ii)))/std(eeg(:,ii));
    %eeg_s = eeg(:,ii);
%     idx = ismember(sensor_labels, lay.label)';
% 
%     tlabels=lay.label(idx)';
%     tpos=lay.pos(idx,:);
    [Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);

    Zi = -replace_nan(Zi);
    
%     eeg_val.Zi = Zi;
%     eeg_val.Yi = Yi;
%     eeg_val.Xi = Xi;
    
    eeg_topos(:,:,ii) = Zi;
    eeg_Xi(:,:,ii) = Xi;
    eeg_Yi(:,:,ii) = Yi;
    
    
    waitbar(ii/n_samples, w_bar, sprintf('Creating CNN input: %d %%', floor(ii/n_samples*100)));
   
end
toc;

close(w_bar);
save('/media/thanos/Elements/thanos/sim_data/sim_type_1/downsampled_dipoles-25k/1e-15/eeg_4TeD_topos.mat', 'eeg_topos', '-v7.3')
save('/media/thanos/Elements/thanos/sim_data/sim_type_1/downsampled_dipoles-25k/1e-15/eeg_4TeD_topos_xi.mat', 'eeg_Xi', '-v7.3')
save('/media/thanos/Elements/thanos/sim_data/sim_type_1/downsampled_dipoles-25k/1e-15/eeg_4TeD_topos_yi.mat', 'eeg_Yi', '-v7.3')


%% visualize

sample = randi([1 n_samples],1,1);
figure;
contourf(eeg_Xi(:,:,sample),eeg_Yi(:,:,sample),eeg_topos(:,:,sample));
colorbar;
title(sprintf('Topography for sample: %d',sample));
