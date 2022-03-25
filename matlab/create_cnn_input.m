clear; close all; clc;

import_fieldtrip();

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

snr='-10'; %sprintf('./../eval_sim_data/two_sources/%sdb/eeg_noisy.npy',snr)
eeg = double(readNPY('./../../../Downloads/test/eeg.npy'));

layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';

[sensors_1010, lay] = compatible_elec(sensor_labels, layout);

n_samples = size(eeg,2);


eeg_topos = zeros(67,67, n_samples);
eeg_Xi = zeros(67,67, n_samples);
eeg_Yi = zeros(67,67, n_samples);

w_bar = waitbar(0, 'Creating CNN input...');

%path_to_save ='../../../Downloads/sim/topos_data/topos/zi_%d.npy';


tic;
for ii=1:n_samples
    eeg_s = (eeg(:,ii) - mean(eeg(:,ii)))/std(eeg(:,ii));
    %eeg_s = eeg(:,ii);

    [Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);

    Zi = -replace_nan(Zi);
    
    %writeNPY(Zi, sprintf(path_to_save,ii))
    
    eeg_topos(:,:,ii) = Zi;
    eeg_Xi(:,:,ii) = Xi;
    eeg_Yi(:,:,ii) = Yi;
    
    
    waitbar(ii/n_samples, w_bar, sprintf('Creating CNN input: %d %%', floor(ii/n_samples*100)));
   
end
toc;

close(w_bar);

% save(sprintf('./../eval_sim_data/one_two_sources/%sdb/eeg_topos_noisy.mat',snr), 'eeg_topos', '-v7.3')
% save(sprintf('./../eval_sim_data/one_two_sources/%sdb/eeg_topos_xi.mat',snr), 'eeg_Xi', '-v7.3')
% save(sprintf('./../eval_sim_data/one_two_sources/%sdb/eeg_topos_yi.mat',snr), 'eeg_Yi', '-v7.3')


save('../../../Downloads/test/eeg_topos.mat', 'eeg_topos', '-v7.3');
save('../../../Downloads/test/eeg_topos_xi.mat', 'eeg_Xi', '-v7.3');
save('../../../Downloads/test/eeg_topos_yi.mat', 'eeg_Yi', '-v7.3');

%% visualize

sample = randi([1 n_samples],1,1);
figure;
contourf(eeg_Xi(:,:,sample),eeg_Yi(:,:,sample),eeg_topos(:,:,sample));
colorbar;
title(sprintf('Topography for sample: %d',sample));
