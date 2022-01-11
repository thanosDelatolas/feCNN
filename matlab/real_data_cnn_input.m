clear; close all; clc;


layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(EEG_avg.label, layout);

% 0.025
eeg_s = EEG_avg.avg(:,151);

[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);
Zi = -replace_nan(Zi);

figure;
contourf(Xi,Yi,Zi)
title('EEG topography.');

save('../../../Downloads/eeg_topo_real.mat', 'Zi');
save('../../../Downloads/eeg_topo_xi_real.mat', 'Xi');
save('../../../Downloads/eeg_topo_yi_real.mat', 'Yi');

%% 


load('../duneuropy/Data/dipoles_downsampled_5k.mat')

pred = readNPY('../../../Downloads/pred_sources.npy');

loc = cd_matrix(:,1:3);

figure;
subplot(1,2,1)
contourf(Xi,Yi,Zi)
title('EEG topography.');



subplot(1,2,2)
scatter3(loc(:,1),loc(:,2),loc(:,3),100,pred,'.')
title('Predicted source');
view([121.7 21.2]);

suptitle('Read Data Prediction')
set(gcf,'Position',[60 180 1600 500])


