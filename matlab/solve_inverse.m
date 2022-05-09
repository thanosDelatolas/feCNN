clear; close all; clc;

%A1999,A1974,A0206
subject='A0206';
%% Load data
load(sprintf('../real_data/%s/EEG_avg.mat',subject));

eeg_idx = get_eeg_idx(subject);

% load downsampled Leadfield
if strcmp(subject,'A0206')
    load('../duneuropy/Data/dipoles.mat')
    % load leadfield (calculated by Duneuro)
    Le = double(readNPY('../duneuropy/DataOut/leadfield.npy'))';
    
    cd_matrix_lt = cd_matrix(:,1:3);
elseif strcmp(subject,'A1974') || strcmp(subject,'A1999')
    % load and resample leadfield
    load(sprintf('../real_data/%s/%s_Le.mat',subject,subject));
    
    load(sprintf('../real_data/%s/%s_source_space.mat',subject,subject));
    
    % apply the linear registration matrix to the cd_matrix
    cd_matrix_lt = apply_lt_matrix(sprintf('../mri_data/%s/%s_regist.mat',subject,subject),cd_matrix(:,1:3));

    
end


% MRI name
T1_name_regist = sprintf('../mri_data/%s/%s_regist_anon.nii',subject,subject);
T1_name = sprintf('../mri_data/%s/%s_anon.nii',subject,subject);



loc_lt = cd_matrix_lt(:,1:3);
loc = cd_matrix(:,1:3);

layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(EEG_avg.label, layout);


eeg_s = (EEG_avg.avg(:,eeg_idx) - mean(EEG_avg.avg(:,eeg_idx)))/std(EEG_avg.avg(:,eeg_idx)) ;
[Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);
Zi = -replace_nan(Zi);

figure;
contourf(Xi,Yi,Zi)
title('EEG topography');

import_directory('./inverse_algorithms/')

%% Single Dipole Scanning
tic;
[dipole_scan_out,best_loc] = SingleDipoleFit(Le, eeg_s);
toc;
[dipole_scan_out,location_dipole_scan] = create_source_activation_vector(...
    dipole_scan_out,'dipole_fit',cd_matrix);

figure;
scatter3(loc(:,1),loc(:,2),loc(:,3),100,dipole_scan_out,'.')
hold on
scatter3(location_dipole_scan(1),location_dipole_scan(2),location_dipole_scan(3),1,max(dipole_scan_out),'y*','linewidth',14);
title('Dipole scanning localization');
hold off;
view([291.3 9.2]);

%% sLORETA

b = eeg_s;
alpha = 20;
tic;
s_loreta_out = sLORETA_with_ori(b,Le,alpha);
toc;
[s_loreta_out,location_sloreta] = create_source_activation_vector(...
    s_loreta_out,'sLORETA',cd_matrix);

figure;
scatter3(loc(:,1),loc(:,2),loc(:,3),100,s_loreta_out,'.')
hold on
scatter3(location_sloreta(1),location_sloreta(2),location_sloreta(3),1,max(s_loreta_out),'y*','linewidth',14);
title('sLORETA localization');
view([291.3 9.2]);


%% Load the neural net's predction

% read neural net's prediction
neural_net_pred = double(readNPY(sprintf('../real_data/%s/pred_sources.npy',subject)));

[neural_net_pred,location_nn] = create_source_activation_vector(...
    neural_net_pred,'loc_cnn',cd_matrix_lt);

figure;
scatter3(loc_lt(:,1),loc_lt(:,2),loc_lt(:,3),100,neural_net_pred,'.')
hold on;
scatter3(location_nn(1),location_nn(2),location_nn(3),100,1,'y*','linewidth',14)
title('Neural Net prediciton');
view([291.3 9.2]);

%% Frobenius norm table

fn_nn_sloreta = norm(location_nn-location_sloreta,'fro');
fn_nn_dipole_fit =  norm(location_nn-location_dipole_scan,'fro');
fn_sloreta_dipole_scan = norm(location_sloreta-location_dipole_scan,'fro');


methods = ["Neural Net vs sLORETA";"Neural Net vs Dipole Scanning";"sLORETA vs Dipole Scanning"];
frobenius_norm = [fn_nn_sloreta;fn_nn_dipole_fit;fn_sloreta_dipole_scan];

res_table = table(methods,frobenius_norm);
disp(res_table)



%% show results on the MRI

import_fieldtrip();
mri_t1_regist        = ft_read_mri(T1_name_regist);
mri_t1        = ft_read_mri(T1_name);

mri_data_scale     = 60;
mri_data_clipping  = 1;

% create the source grid
source_grid_lt = downsample(cd_matrix_lt(:,1:3),3);
source_grid = downsample(cd_matrix(:,1:3),3);

% project to MRI the neural net's prediction
source_activation_mri_alt(mri_t1_regist,mri_data_scale,neural_net_pred,source_grid_lt,...
    mri_data_clipping,EEG_avg.time(eeg_idx),'Localization with Neural Net');

% project to MRI the sLORETA solution
source_activation_mri_alt(mri_t1,mri_data_scale,s_loreta_out,source_grid,...
    mri_data_clipping,EEG_avg.time(eeg_idx),'Localization with sLORETA');


% project to MRI the dipole scan solution
source_activation_mri_alt(mri_t1,mri_data_scale,dipole_scan_out,source_grid,...
    mri_data_clipping,EEG_avg.time(eeg_idx),'Localization with Dipole Scan');


%% Save files

%path_to_save = '../../GitHub/thesis_summary/pics/';
%saver(path_to_save,2, 3, 0);
