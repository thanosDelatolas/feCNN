clear; close all; clc;


%Le = double(readNPY('../duneuropy/DataOut/leadfield.npy'))';
%load('../duneuropy/Data/dipoles.mat')

% load downsampled Leadfield and dipoles
Le = double(readNPY('../duneuropy/DataOut/leadfield_downsampled_10k.npy'))';
load('../duneuropy/Data/dipoles_downsampled_10k.mat')

% load the real data
load('../real_data/EEG_avg.mat')

layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(EEG_avg.label, layout);
eeg_s = (EEG_avg.avg(:,151) - mean(EEG_avg.avg(:,151)))/std(EEG_avg.avg(:,151)) ;


[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);
Zi = -replace_nan(Zi);

import_directory('./inverse_algorithms/')
%% Single Dipole Fit

[dip,best_loc] = SingleDipoleFit(Le, eeg_s);

[val,idx_max] = max(dip);

loc = cd_matrix(:,1:3);
figure;
subplot(1,2,1)
contourf(Xi,Yi,Zi)
title('EEG topography.');

subplot(1,2,2)
scatter3(loc(:,1),loc(:,2),loc(:,3),100,dip,'.')
hold on
scatter3(loc(idx_max,1),loc(idx_max,2),loc(idx_max,3),1,dip(idx_max),'y', 'linewidth',22)
title('Dipole fitting localization');
view([-85.9 11.6]);
colorbar;

set(gcf,'Position',[60 180 1600 500])

%% sLORETA

b = eeg_s;
alpha = 25;
[u_sLORETA,s] = sLORETA_dir(b,Le,alpha);



