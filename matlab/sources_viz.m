%% Load data
clear; close all; clc;

load('../duneuropy/Data/dipoles.mat')

sources = readNPY('/home/thanos/Downloads/sources.npy');
%predicted_sources = readNPY('/home/thanos/Downloads/predicted_sources.npy');

load('/home/thanos/Downloads/eeg_topos.mat')
load('/home/thanos/Downloads/eeg_topos_xi.mat')
load('/home/thanos/Downloads/eeg_topos_yi.mat')

%% visualize

% close all;

n_samples = size(sources,2);

source_idx = randi([1 n_samples],1,1);


source = sources(:, source_idx);
% pred = predicted_sources(:,source_idx);
loc = cd_matrix(:,1:3);


figure;
subplot(1,3,1)
scatter3(loc(:,1),loc(:,2),loc(:,3),100,source,'.')
title('Simulated source');

% subplot(1,3,2)
% scatter3(X,Y,Z,100,pred,'.')
% title('Predicted source');


Zi = eeg_topos(:,:,source_idx);
Xi = eeg_Xi(:,:,source_idx);
Yi = eeg_Yi(:,:,source_idx);

subplot(1,3,3)
contourf(Xi,Yi,Zi)
title('EEG topography.');
suptitle(sprintf('Source %d',source_idx))

