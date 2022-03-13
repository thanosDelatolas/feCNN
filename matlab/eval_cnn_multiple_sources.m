clear; close all; clc;

% This file evaluates our cnn with the noisy simulated eeg data (multiple
% sources)

Le = double(readNPY('../duneuropy/DataOut/leadfield_downsampled_10k.npy'))';
load('../duneuropy/Data/dipoles_downsampled_10k.mat')
loc = cd_matrix(:,1:3);

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);


layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(sensor_labels, layout);

snr = int2str(20);


eeg_signals = double(readNPY(sprintf('./../eval_sim_data/two_sources/%sdb/eeg_noisy.npy',snr)));
% ground truth
sources_val = double(readNPY(sprintf('./../eval_sim_data/two_sources/%sdb/sources.npy',snr)));
source_centers = readNPY(sprintf('./../eval_sim_data/two_sources/%sdb/source_centers.npy',snr));

sample = 10;

eeg = eeg_signals(:,sample);
centers = source_centers(sample,:);
source = sources_val(:,sample);

locations = find_multiple_soucres(source,cd_matrix);
