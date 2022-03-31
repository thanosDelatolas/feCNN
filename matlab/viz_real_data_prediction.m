clear; close all; clc;

%A1999,A1974,A0206
subject='A1999';
ms='22_5';
load(sprintf('../real_data/%s/EEG_avg.mat',subject));

% plot EEG_avg
pol = -1;     % correct polarity
scale = 10^6; % scale for eeg data micro volts
signal_EEG = scale*pol*EEG_avg.avg; % add single trials in a new value
figure;
plot(EEG_avg.time,signal_EEG,'color',[0,0,0.5]);


load(sprintf('../real_data/%s/%sms/eeg_topo_real_%sms.mat',subject,ms,ms));
load(sprintf('../real_data/%s/%sms/eeg_topo_real_xi_%sms.mat',subject,ms,ms));
load(sprintf('../real_data/%s/%sms/eeg_topo_real_yi_%sms.mat',subject,ms,ms));



% load source space and the fsl linear registration output.
if strcmp(subject,'A1974')
    load(sprintf('../real_data/%s/%s_source_space.mat',subject,subject));
    cd_matrix = apply_lt_matrix(sprintf('../mri_data/%s/%s_regist.mat',subject,subject),cd_matrix(:,1:3));
    % downsample the source_space
    len = size(cd_matrix,1);
    cd_matrix = resample(cd_matrix,10092,len);
    
    eeg_idx = 149;
elseif strcmp(subject,'A1999')
    load(sprintf('../real_data/%s/%s_source_space.mat',subject,subject));
    cd_matrix = apply_lt_matrix(sprintf('../mri_data/%s/%s_regist.mat',subject,subject),cd_matrix(:,1:3));
    % downsample the source_space
    len = size(cd_matrix,1);
    cd_matrix = resample(cd_matrix,10092,len);
    
     eeg_idx = 148;
elseif strcmp(subject,'A0206')
    load('../duneuropy/Data/dipoles_downsampled_10k.mat')
    
    if strcmp(ms,'20')
            eeg_idx = 145;
    elseif strcmp(ms,'24_2')
            eeg_idx = 150;
    elseif strcmp(ms,'25')
        eeg_idx = 151;
    elseif  strcmp(ms,'25_8')
        eeg_idx = 152;
    end
end

% load the neural net's prediction

neural_net_pred = readNPY(sprintf('../real_data/%s/%sms/pred_sources_%s.npy',subject,ms,ms));


[neural_net_pred,location_nn] = create_source_activation_vector(...
    neural_net_pred,'nn',cd_matrix);

figure;
scatter3(cd_matrix(:,1),cd_matrix(:,2),cd_matrix(:,3),100,neural_net_pred,'.')
title('Neural Net prediciton');


import_fieldtrip();
T1_name = sprintf('../mri_data/%s/%s_regist.nii',subject,subject);
mri_t1        = ft_read_mri(T1_name);

mri_data_scale     = 60;
mri_data_clipping  = .8;

% create the source grid
source_grid = downsample(cd_matrix(:,1:3),3);


% project to MRI the neural net's prediction
source_activation_mri(mri_t1,mri_data_scale,neural_net_pred,source_grid,...
    mri_data_clipping,EEG_avg.time(eeg_idx),'Localization with Neural Net');

