clear; close all; clc;

import_fieldtrip();

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

eeg = double(readNPY('/media/thanos/Big Data/Thanos/TUC/Thesis/sim_data/sim_type_1/sim_100k_eeg.npy'));

% sources = double(readNPY('/data/sources.npy'));

layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';

[sensors_1010, lay] = compatible_elec(sensor_labels, layout);

n_samples = size(eeg,2);

eeg_arr = zeros(67,67, n_samples);
% eeg_Xi = zeros(67,67, n_samples);
% eeg_Yi = zeros(67,67, n_samples);

w_bar = waitbar(0, 'Creating CNN input...');

tic;
for ii=1:n_samples
    eeg_s = eeg(:,ii);

    idx = ismember(sensor_labels, lay.label)';

    tlabels=lay.label(idx)';
    tpos=lay.pos(idx,:);
    [Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);

    Zi = replace_nan(Zi);
    
%     eeg_val.Zi = Zi;
%     eeg_val.Yi = Yi;
%     eeg_val.Xi = Xi;
    
    eeg_arr(:,:,ii) = Zi;
%     eeg_Xi(:,:,ii) = Xi;
%     eeg_Yi(:,:,ii) = Yi;
    
    
    waitbar(ii/n_samples, w_bar, sprintf('Creating CNN input: %d %%', floor(ii/n_samples*100)));
    %pause(0.1);
end
toc;

close(w_bar);
save('eeg_topographies.mat', 'eeg_arr', '-v7.3')
