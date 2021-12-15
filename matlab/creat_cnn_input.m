clear; close all; clc;

import_fieldtrip();

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

eeg = double(readNPY('/home/thanos/Downloads/eeg.npy'));

% sources = double(readNPY('/data/sources.npy'));

layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';

[sensors_1010, lay] = compatible_elec(sensor_labels, layout);

n_samples = size(eeg,2);

eeg_topos = zeros(67,67, n_samples);
eeg_Xi = zeros(67,67, n_samples);
eeg_Yi = zeros(67,67, n_samples);

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
    
    eeg_topos(:,:,ii) = Zi;
    eeg_Xi(:,:,ii) = Xi;
    eeg_Yi(:,:,ii) = Yi;
    
    
    waitbar(ii/n_samples, w_bar, sprintf('Creating CNN input: %d %%', floor(ii/n_samples*100)));
    %pause(0.1);
end
toc;

close(w_bar);
save('/home/thanos/Downloads/eeg_topos', 'eeg_topos', '-v7.3')
save('/home/thanos/Downloads/eeg_topos_xi', 'eeg_Xi', '-v7.3')
save('/home/thanos/Downloads/eeg_topos_yi', 'eeg_Yi', '-v7.3')
