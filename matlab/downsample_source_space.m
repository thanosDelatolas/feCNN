clear; close all; clc;

load('../duneuropy/Data/dipoles.mat');
dipoles_downsampled = downsample(cd_matrix,5);

cd_matrix = dipoles_downsampled;

save('../duneuropy/Data/dipoles_downsampled_10k.mat', 'cd_matrix');