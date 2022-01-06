%% Load data
clear; close all; clc;

load('../duneuropy/Data/dipoles.mat')

sources = readNPY('/media/thanos/Elements/thanos/sim_data/sim_type_1/region_dataset/Region 8/region_8_sources.npy');
predicted_sources = readNPY('../../../Downloads/pred_sources.npy');

load('/media/thanos/Elements/thanos/sim_data/sim_type_1/region_dataset/Region 8/region_8_eeg_topos.mat')
load('/media/thanos/Elements/thanos/sim_data/sim_type_1/region_dataset/Region 8/region_8_eeg_topos_xi.mat')
load('/media/thanos/Elements/thanos/sim_data/sim_type_1/region_dataset/Region 8/region_8_eeg_topos_yi.mat')


%% visualize

first_dipole=34900;
last_dipole=40100;
% close all;

n_samples = size(predicted_sources,2);

source_idx = randi([1 n_samples],1,1);


source = sources(:, source_idx);
pred = predicted_sources(:,source_idx);
loc = cd_matrix(:,1:3);

Zi = eeg_topos(:,:,source_idx);
Xi = eeg_Xi(:,:,source_idx);
Yi = eeg_Yi(:,:,source_idx);

figure;
subplot(1,3,1)
contourf(Xi,Yi,Zi)
title('EEG topography.');


subplot(1,3,2)
scatter3(loc(first_dipole:last_dipole-1,1),loc(first_dipole:last_dipole-1,2),loc(first_dipole:last_dipole-1,3),5200,source,'.')
title('Simulated source');
view([121.7 21.2]);

subplot(1,3,3)
scatter3(loc(first_dipole:last_dipole-1,1),loc(first_dipole:last_dipole-1,2),loc(first_dipole:last_dipole-1,3),5200,pred,'.')
title('Predicted source');
view([121.7 21.2]);

suptitle(sprintf('Sample %d',source_idx))
set(gcf,'Position',[60 180 1600 500])
