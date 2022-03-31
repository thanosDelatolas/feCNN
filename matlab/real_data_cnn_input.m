clear; close all; clc;

%A1999,A1974,A0206
subject='A1974';
load(sprintf('../real_data/%s/EEG_avg.mat',subject));

% plot EEG_avg
pol = -1;     % correct polarity
scale = 10^6; % scale for eeg data micro volts
signal_EEG = scale*pol*EEG_avg.avg; % add single trials in a new value
figure;
plot(EEG_avg.time,signal_EEG,'color',[0,0,0.5]);


layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(EEG_avg.label, layout);


 ms = '25';
if strcmp(subject,'A0206')
    % 20 ms 145 (deep source)
    % 24.2 ms 150
    % 25ms ,151
    % 25.8 ms 152
    if strcmp(ms,'20')
            idx = 145;
    elseif strcmp(ms,'24_2')
            idx = 150;
    elseif strcmp(ms,'25')
        idx = 151;
    elseif  strcmp(ms,'25_8')
        idx = 152;
    end
elseif strcmp(subject,'A1974')
    % 23.3 ms
    idx = 149;
    ms='23_3';
elseif strcmp(subject,'A1999')
    % 22.5 ms
%     if strcmp(ms,'22_5')
%         idx=148;
%     end
    idx = 148;
    ms='22_5';
end


eeg_s = (EEG_avg.avg(:,idx) - mean(EEG_avg.avg(:,idx)))/std(EEG_avg.avg(:,idx));
[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);
Zi = -replace_nan(Zi);

figure;
contourf(Xi,Yi,Zi)
title('EEG topography.');

save(sprintf('../real_data/%s/%sms/eeg_topo_real_%sms.mat',subject,ms,ms), 'Zi');
save(sprintf('../real_data/%s/%sms/eeg_topo_real_xi_%sms.mat',subject,ms,ms), 'Xi');
save(sprintf('../real_data/%s/%sms/eeg_topo_real_yi_%sms.mat',subject,ms,ms), 'Yi');

%% 

load(sprintf('../real_data/%s/%sms/eeg_topo_real_%sms.mat',subject,ms,ms));
load(sprintf('../real_data/%s/%sms/eeg_topo_real_xi_%sms.mat',subject,ms,ms));
load(sprintf('../real_data/%s/%sms/eeg_topo_real_yi_%sms.mat',subject,ms,ms));



pred = readNPY(sprintf('../real_data/%s/%sms/pred_sources_%s.npy',subject,ms,ms));
[~,idx] = max(pred);


load(sprintf('../real_data/%s/%s_source_space.mat',subject,subject));
cd_matrix = apply_lt_matrix(sprintf('../mri_data/%s/%s_regist.mat',subject,subject),cd_matrix(:,1:3));
% downsample the source_space
len = size(cd_matrix,1);
cd_matrix = resample(cd_matrix,10092,len);

location = cd_matrix(idx,:);
loc = cd_matrix(:,1:3);

figure;
subplot(1,2,1)
contourf(Xi,Yi,Zi)
title('EEG topography.');

subplot(1,2,2)
scatter3(loc(:,1),loc(:,2),loc(:,3),100,pred,'.')
view([121.7 21.2]);
title('Pred with new cnn');

%suptitle(strrep(sprintf('Read Data Prediction %sms',ms),'_','.'));
set(gcf,'Position',[60 180 1600 500])



