clear; close all; clc;

import_fieldtrip();

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

eeg = double(readNPY('/media/thanos/Big Data/Thanos/TUC/Thesis/sim_data/sim_type_2/eeg_big_sim_1.npy'));

% sources = double(readNPY('/data/sources.npy'));

layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';

[sensors_1010, lay] = compatible_elec(sensor_labels, layout);


eeg_s = eeg(:,7);
% scatter3(sensors(:,1),sensors(:,2),sensors(:,3),100,eeg_s,'.')

idx = ismember(sensor_labels, lay.label)';

tlabels=lay.label(idx)';
tpos=lay.pos(idx,:);
[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);

%Zi = replace_nan(Zi);
figure;
fac = 0.9;
contourf(Xi,Yi,Zi);
hold on;
scatter(sensors_1010(:,1),sensors_1010(:,2),100,'k','.');
hold on;
plot(lay.outline{1}(:,1)*fac,lay.outline{1}(:,2)*fac,'k');
plot(lay.outline{2}(:,1)*fac,lay.outline{2}(:,2)*fac,'k');
plot(lay.outline{3}(:,1)*fac,lay.outline{3}(:,2)*fac,'k');
plot(lay.outline{4}(:,1)*fac,lay.outline{4}(:,2)*fac,'k');
title('ft\_plot\_topo');
colorbar;
%saveas(gcf,'../assets/sim.png')

% figure; ft_plot_topo3d(sensors,eeg_s); title('ft\_plot\_topo3d'); colorbar;