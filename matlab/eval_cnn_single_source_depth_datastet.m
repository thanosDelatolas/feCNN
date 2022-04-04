clear; close all; clc;
% This file evaluates our cnn with the noisy simulated eeg data (single
% source)

Le = double(readNPY('../duneuropy/DataOut/leadfield.npy'))';
load('../duneuropy/Data/dipoles.mat')
loc = cd_matrix(:,1:3);

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);


layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(sensor_labels, layout);


import_directory('./inverse_algorithms/');

snr_db = '-10';

depth_dataset_path = sprintf('../eval_sim_data/depth/%sdb/',snr_db);
depths_struct = dir(fullfile(depth_dataset_path,'*'));

w_bar_title = sprintf('Evaluating depth dataset for snr=%sdB',snr_db);
w_bar = waitbar(0,w_bar_title);
% list of subfolders of depth_dataset_path.
depths_subdirs = setdiff({depths_struct([depths_struct.isdir]).name},{'.','..'});

% vectors with the Mean Localization Erro per snr
mle_cnn = zeros(size(depths_subdirs));
mle_s_loreta = zeros(size(depths_subdirs));
mle_dipole_scan = zeros(size(depths_subdirs));
depth = zeros(size(depths_subdirs));
tic;
for ii = 1:numel(depths_subdirs)
    % file extention or *
    subdir = dir(fullfile(depth_dataset_path,depths_subdirs{ii},'*'));
    C = {subdir(~[subdir.isdir]).name}; % files in subfolder.
    
    % read the eeg and both th simulated and predicted (from the cnn) sources
    for jj = 1:numel(C)
        if strcmp(C{jj},'eeg.npy')
            eeg_signals = double(readNPY(fullfile(depth_dataset_path,depths_subdirs{ii},C{jj})));
        elseif strcmp(C{jj},'sources.npy')
            simulated_sources = double(readNPY(fullfile(depth_dataset_path,depths_subdirs{ii},C{jj})));
        elseif strcmp(C{jj},'predicted_sources.npy')
            cnn_predictions = double(readNPY(fullfile(depth_dataset_path,depths_subdirs{ii},C{jj})));
        elseif strcmp(C{jj},'source_centers.npy')
            source_centers = double(readNPY(fullfile(depth_dataset_path,depths_subdirs{ii},C{jj})));
        end
    end
    
    % evaluate for this depth (depths_subdirs{ii})
    n_samples = size(eeg_signals,2);
    
    le_cnn = zeros(n_samples,1);
    le_s_loreta = zeros(n_samples,1);
    le_dipole_scan = zeros(n_samples,1);
    
    for kk=1:n_samples
       
        
        % noisy eeg signal for sample kk
        eeg_s = eeg_signals(:,kk);
        % ground truth for sample kk
        source =  source_centers(kk)+1;        
        location = cd_matrix(source,1:3);
        
        % prediction of th cnn
        cnn_pred = cnn_predictions(kk);        
        
        % dipole scanning
        [dipole_scan_out,best_loc] = SingleDipoleFit(Le, eeg_s);        
        [dipole_scan_out,location_dipole_scan] = create_source_activation_vector(...
            dipole_scan_out,'dipole_fit',cd_matrix);
        
        % sLORETA
        s_loreta_out = sLORETA_with_ori(eeg_s,Le,25);
        [s_loreta_out,location_sloreta] = create_source_activation_vector(...
            s_loreta_out,'sLORETA',cd_matrix);
        
        le_cnn(kk) = distance_3d_space(location, cnn_pred);
        le_s_loreta(kk) = distance_3d_space(location, location_sloreta);
        le_dipole_scan(kk) = distance_3d_space(location,location_dipole_scan);
        
        
    end
    
    mle_cnn(ii) = mean(le_cnn);
    mle_s_loreta(ii) = mean(le_s_loreta);
    mle_dipole_scan(ii) = mean(le_dipole_scan);
    depth(ii) = str2double(depths_subdirs{ii});
    waitbar(ii/numel(depths_subdirs), w_bar, strcat(w_bar_title,sprintf(':%d %%', floor(ii/numel(depths_subdirs)*100))));
end
 save(sprintf('./eval_results/depth_eval/%sdb/mle_cnn.mat',snr_db),'mle_cnn');
 save(sprintf('./eval_results/depth_eval/%sdb/mle_s_loreta.mat',snr_db),'mle_s_loreta');
 save(sprintf('./eval_results/depth_eval/%sdb/mle_dipole_scan.mat',snr_db),'mle_dipole_scan');
 save('./eval_results/depth_eval/depth.mat','depth');
 
close(w_bar);

toc;

%%
snr_db='-5';

load(sprintf('./eval_results/depth_eval/%sdb/mle_cnn.mat',snr_db));
load(sprintf('./eval_results/depth_eval/%sdb/mle_s_loreta.mat',snr_db));
load(sprintf('./eval_results/depth_eval/%sdb/mle_dipole_scan.mat',snr_db));
load('./eval_results/depth_eval/depth.mat');

figure;
plot(sort(depth),sort(mle_cnn),'linewidth',4);
hold on;
plot(sort(depth),sort(mle_s_loreta),'linewidth',4);
hold on;
plot(sort(depth),sort(mle_dipole_scan),'linewidth',4);
hold off;
grid on;
legend('CNN','Dipole Scan', 'sLORETA');
set(gcf,'Position',[220 300 1200 500]);
xlabel('Depth [mm]');
ylabel('Localization Error [mm]');