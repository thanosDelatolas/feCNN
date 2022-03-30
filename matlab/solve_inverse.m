clear; close all; clc;


%Le = double(readNPY('../duneuropy/DataOut/leadfield.npy'))';
%load('../duneuropy/Data/dipoles.mat')

% load downsampled Leadfield and dipoles
Le = double(readNPY('../duneuropy/DataOut/leadfield_downsampled_10k.npy'))';
load('../duneuropy/Data/dipoles_downsampled_10k.mat')
loc = cd_matrix(:,1:3);

% load the real data
load('../real_data/EEG_avg.mat')

layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(EEG_avg.label, layout);


ms = '25';
if strcmp(ms,'20')
        eeg_idx = 145;
elseif strcmp(ms,'24_2')
        eeg_idx = 150;
elseif strcmp(ms,'25')
    eeg_idx = 151;
elseif  strcmp(ms,'25_8')
    eeg_idx = 152;
end
eeg_s = (EEG_avg.avg(:,eeg_idx) - mean(EEG_avg.avg(:,eeg_idx)))/std(EEG_avg.avg(:,eeg_idx)) ;
[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);
Zi = -replace_nan(Zi);

figure;
contourf(Xi,Yi,Zi)
title(sprintf('EEG topography at %s ms',ms));

import_directory('./inverse_algorithms/')

%% Single Dipole Scanning

[dipole_scan_out,best_loc] = SingleDipoleFit(Le, eeg_s);

[dipole_scan_out,location_dipole_scan] = create_source_activation_vector(...
    dipole_scan_out,'dipole_fit',cd_matrix);

figure;
scatter3(loc(:,1),loc(:,2),loc(:,3),100,dipole_scan_out,'.')

hold on
scatter3(location_dipole_scan(1),location_dipole_scan(2),location_dipole_scan(3),1,max(dipole_scan_out),'y','linewidth',9);
title('Dipole scanning localization');

if strcmp(ms,'25')
    view([291.3 9.2]);
else
    view([-251.1 7.6]);
end


%% sLORETA

b = eeg_s;
alpha = 25;
s_loreta_out = sLORETA_with_ori(b,Le,alpha);

[s_loreta_out,location_sloreta] = create_source_activation_vector(...
    s_loreta_out,'sLORETA',cd_matrix);

figure;
scatter3(loc(:,1),loc(:,2),loc(:,3),100,s_loreta_out,'.')
hold on
scatter3(location_sloreta(1),location_sloreta(2),location_sloreta(3),1,max(s_loreta_out),'y','linewidth',9);
title('sLORETA localization');

if strcmp(ms,'25')
    view([291.3 9.2]);
else
    view([-251.1 7.6]);
end


%% Load the neural net's predction

% read neural net's prediction
neural_net_pred = double(readNPY(sprintf('../real_data/%sms/pred_sources_%s.npy',ms,ms)));

[neural_net_pred,location_nn] = create_source_activation_vector(...
    neural_net_pred,'nn',cd_matrix);

figure;
scatter3(loc(:,1),loc(:,2),loc(:,3),100,neural_net_pred,'.')
title('Neural Net prediciton');
if strcmp(ms,'25')
    view([291.3 9.2]);
else
    view([-251.1 7.6]);
end

%% Frobenius norm table

fn_nn_sloreta = norm(location_nn-location_sloreta,'fro');
fn_nn_dipole_fit =  norm(location_nn-location_dipole_scan,'fro');
fn_sloreta_dipole_scan = norm(location_sloreta-location_dipole_scan,'fro');


methods = ["Neural Net vs sLORETA";"Neural Net vs Dipole Scanning";"sLORETA vs Dipole Scanning"];
frobenius_norm = [fn_nn_sloreta;fn_nn_dipole_fit;fn_sloreta_dipole_scan];

res_table = table(methods,frobenius_norm);
disp(res_table)

%% Spatial Dispersion (SD)



%% show results on the MRI

import_fieldtrip();
T1_name = '/media/thanos/TD/A0206/o20170327_111915t1mpragesagiso1mmwselnfp2s003a1001.nii';%'../mri_data/T1w_1mm_anon.nii';
mri_t1        = ft_read_mri(T1_name);

mri_data_scale     = 60;
mri_data_clipping  = .8;

% create the source grid
source_grid = downsample(cd_matrix(:,1:3),3);


% project to MRI the neural net's prediction
source_activation_mri(mri_t1,mri_data_scale,neural_net_pred,source_grid,...
    mri_data_clipping,EEG_avg.time(eeg_idx),'Localization with Neural Net');

% project to MRI the sLORETA solution
source_activation_mri(mri_t1,mri_data_scale,s_loreta_out,source_grid,...
    mri_data_clipping,EEG_avg.time(eeg_idx),'Localization with sLORETA');


% project to MRI the dipole fit solution
source_activation_mri(mri_t1,mri_data_scale,dipole_scan_out,source_grid,...
    mri_data_clipping,EEG_avg.time(eeg_idx),'Localization with Dipole Scan');


%% Save files

path_to_save = '../../GitHub/thesis_summary/pics/';
%saver(path_to_save,2, 3, 0);
