clear; close all; clc;

load('../duneuropy/Data/dipoles.mat');
dipoles_downsampled = downsample(cd_matrix,2);

cd_matrix = dipoles_downsampled;

save('../duneuropy/Data/dipoles_downsampled_25k.mat', 'cd_matrix');