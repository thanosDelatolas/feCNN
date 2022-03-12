clear; close all; clc;


load('../real_data/EEG_avg.mat')
layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(EEG_avg.label, layout);

% 20 ms 145 (deep source_
% 24.2 ms 150
% 25ms ,151
% 25.8 ms 152

ms = '20';
if strcmp(ms,'20')
        idx = 145;
elseif strcmp(ms,'24_2')
        idx = 150;
elseif strcmp(ms,'25')
    idx = 151;
elseif  strcmp(ms,'25_8')
    idx = 152;
end

eeg_s = (EEG_avg.avg(:,idx) - mean(EEG_avg.avg(:,idx)))/std(EEG_avg.avg(:,idx));
[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);
Zi = -replace_nan(Zi);

figure;
contourf(Xi,Yi,Zi)
title('EEG topography.');

save(sprintf('../real_data/%sms/eeg_topo_real_%sms.mat',ms,ms), 'Zi');
save(sprintf('../real_data/%sms/eeg_topo_real_xi_%sms.mat',ms,ms), 'Xi');
save(sprintf('../real_data/%sms/eeg_topo_real_yi_%sms.mat',ms,ms), 'Yi');

%% 

ms='20';

load(sprintf('../real_data/%sms/eeg_topo_real_%sms.mat',ms,ms));
load(sprintf('../real_data/%sms/eeg_topo_real_xi_%sms.mat',ms,ms));
load(sprintf('../real_data/%sms/eeg_topo_real_yi_%sms.mat',ms,ms));


load('../duneuropy/Data/dipoles_downsampled_10k.mat')

pred = readNPY(sprintf('../real_data/%sms/pred_sources_%s.npy',ms,ms));
[~,idx] = max(pred);
location = cd_matrix(idx,:);

loc = cd_matrix(:,1:3);

figure;
subplot(1,3,1)
contourf(Xi,Yi,Zi)
title('EEG topography.');


subplot(1,3,2)
scatter3(loc(:,1),loc(:,2),loc(:,3),100,pred,'.')
title('Predicted source');
view([121.7 21.2]);

subplot(1,3,3)
pred_new = readNPY('../../../Downloads/pred_real.npy');
scatter3(loc(:,1),loc(:,2),loc(:,3),100,pred,'.')
view([121.7 21.2]);
title('Pred with new cnn');

%suptitle(strrep(sprintf('Read Data Prediction %sms',ms),'_','.'));
set(gcf,'Position',[60 180 1600 500])



