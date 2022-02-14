clear; close all; clc;


load('../real_data/EEG_avg.mat')
layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(EEG_avg.label, layout);

% 25ms ,151
% 24.2 ms 150
% 25.8 ms 152
% 24 ms 145

eeg_s = (EEG_avg.avg(:,152) - mean(EEG_avg.avg(:,152)))/std(EEG_avg.avg(:,152));
[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);
Zi = -replace_nan(Zi);

figure;
contourf(Xi,Yi,Zi)
title('EEG topography.');

save('../real_data/eeg_topo_real_25_8ms.mat', 'Zi');
save('../real_data/eeg_topo_real_xi_25_8ms.mat', 'Xi');
save('../real_data/eeg_topo_real_yi_25_8ms.mat', 'Yi');

%% 

ms='25';

load(sprintf('../real_data/%sms/eeg_topo_real_%sms.mat',ms,ms));
load(sprintf('../real_data/%sms/eeg_topo_real_xi_%sms.mat',ms,ms));
load(sprintf('../real_data/%sms/eeg_topo_real_yi_%sms.mat',ms,ms));


load('../duneuropy/Data/dipoles.mat')

%pred = readNPY('../real_data/25_8ms/pred_sources_25_8.npy');
pred = readNPY('../../../Downloads/pred_sources_real.npy');



loc = cd_matrix(:,1:3);

figure;
subplot(1,2,1)
contourf(Xi,Yi,Zi)
title('EEG topography.');


subplot(1,2,2)
scatter3(loc(:,1),loc(:,2),loc(:,3),100,pred,'.')

title('Predicted source');
view([121.7 21.2]);

%suptitle(strrep(sprintf('Read Data Prediction %sms',ms),'_','.'));
set(gcf,'Position',[60 180 1600 500])


