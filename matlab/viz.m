clear; close all; clc;

import_fieldtrip();

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

eeg = double(readNPY('/home/thanos/Downloads/eeg.npy'));

sources = double(readNPY('/home/thanos/Downloads/sources.npy'));

le = double(readNPY('../duneuropy/DataOut/leadfield.npy'));
% 
% eeg_m = le' * sources;


layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';

[sensors_1010, lay] = compatible_elec(sensor_labels, layout);

n_samples = size(eeg,2);
sample = randi([1 n_samples],1,1);

%sample = 10;

eeg_s = eeg(:,sample);

% eeg_m_s = eeg_m(:, sample);
% scatter3(sensors(:,1),sensors(:,2),sensors(:,3),73,eeg_s,'.')

idx = ismember(sensor_labels, lay.label)';

tlabels=lay.label(idx)';
tpos=lay.pos(idx,:);

[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);


Zi = replace_nan(Zi);
figure;
subplot(1,2,1)
fac = 0.9;
contourf(Xi,Yi,-Zi);
hold on;
scatter(sensors_1010(:,1),sensors_1010(:,2),100,'k','.');
hold on;
plot(lay.outline{1}(:,1)*fac,lay.outline{1}(:,2)*fac,'k');
plot(lay.outline{2}(:,1)*fac,lay.outline{2}(:,2)*fac,'k');
plot(lay.outline{3}(:,1)*fac,lay.outline{3}(:,2)*fac,'k');
plot(lay.outline{4}(:,1)*fac,lay.outline{4}(:,2)*fac,'k');
title(sprintf('Topography for sample: %d',sample));
colorbar;
%saveas(gcf,'../assets/sim.png')

% figure; ft_plot_topo3d(sensors,eeg_s); title('ft\_plot\_topo3d'); colorbar;

load('../duneuropy/Data/dipoles.mat')
source = sources(:, sample);
loc = cd_matrix(:,1:3);
subplot(1,2,2);
scatter3(loc(:,1),loc(:,2),loc(:,3),100,source,'.')
title(sprintf('Simulated source space for sample: %d',sample));
colorbar;
view([-103.9 -6.8])
set(gcf,'Position',[60 180 1600 500])


