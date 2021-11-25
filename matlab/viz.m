clear; close all; clc;


import_fieldtrip();


[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

eeg = readNPY('./data/eeg.npy');


layout = '/home/thanos/fieldtrip/template/layout/elec1010.lay';

[sensors_1010, lay] = compatible_elec(sensor_labels, layout);


eeg_s = eeg(:,1);
% scatter3(sensors(:,1),sensors(:,2),sensors(:,3),100,eeg_s,'.')

idx = ismember(sensor_labels, lay.label)';

tlabels=lay.label(idx)';
tpos=lay.pos(idx,:);
figure; ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline); colorbar;
title('ft\_plot\_topo');

figure; ft_plot_topo3d(sensors,eeg_s); title('ft\_plot\_topo3d'); colorbar;