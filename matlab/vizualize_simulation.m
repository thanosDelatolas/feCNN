clear all;
close all;
clc;


restoredefaultpath % restore default folder for matlab
maindir = '/home/thanos/thesis/matlab';     % keep main path

% set up the path of fieldtrip
cd('/home/thanos/fieldtrip')
addpath('/home/thanos/fieldtrip')
ft_defaults
cd(maindir)


[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

eeg = readNPY('./data/eeg.npy');
leadfield = readNPY('./../duneuropy/DataOut/leadfield.npy')';
simulated_sources = readNPY('./data/sources.npy');


% J1 = simulated_sources(:,1,1);
% eeg_s=leadfield*J1;
% scatter3(sensors(:,1),sensors(:,2),sensors(:,3),100,eeg_s,'.')

eeg_s = eeg(:,1);


cfg=[];
cfg.layout= '/home/thanos/fieldtrip/template/layout/elec1010.lay';
lay=ft_prepare_layout(cfg);
% figure; ft_plot_layout(lay);


idx = ismember(sensor_labels, lay.label)';
tlabels=lay.label(idx)';
tpos=lay.pos(idx,:);
% ft_plot_topo(tpos(:,1),tpos(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);

%figure; ft_plot_topo3d(sensors,eeg_s);

