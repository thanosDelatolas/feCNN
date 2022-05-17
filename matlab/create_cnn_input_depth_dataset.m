clear; close all; clc;

import_fieldtrip();

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);
layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(sensor_labels, layout);

snr='20';
depth_dataset_path = sprintf('../eval_sim_data/old_nn/depth/%sdb/',snr);
depths_struct = dir(fullfile(depth_dataset_path,'*'));

w_bar_title = sprintf('CNN input for depth dataset with snr=%sdB',snr);
w_bar = waitbar(0,w_bar_title);
% list of subfolders of depth_dataset_path.
depths_subdirs = setdiff({depths_struct([depths_struct.isdir]).name},{'.','..'});
for ii = 1:numel(depths_subdirs)
    % file extention or *
    subdir = dir(fullfile(depth_dataset_path,depths_subdirs{ii},'*'));
    C = {subdir(~[subdir.isdir]).name}; % files in subfolder.
    
    % read the egg data
    for jj = 1:numel(C)
        if strcmp(C{jj},'eeg.npy')
            eeg = double(readNPY(fullfile(depth_dataset_path,depths_subdirs{ii},C{jj})));
            break
        end
    end
    
    % create topos
    n_samples = size(eeg,2);
    eeg_topos = zeros(67,67, n_samples);
    eeg_Xi = zeros(67,67, n_samples);
    eeg_Yi = zeros(67,67, n_samples);
    for kk=1:n_samples
        eeg_s = (eeg(:,kk) - mean(eeg(:,kk)))/std(eeg(:,kk));
        [Zi, Yi, Xi ] = ft_plot_topo(sensors_1010(:,1),sensors_1010(:,2),eeg_s,'mask',lay.mask,'outline',lay.outline);

        Zi = -replace_nan(Zi);
        
        eeg_topos(:,:,kk) = Zi;
        eeg_Xi(:,:,kk) = Xi;
        eeg_Yi(:,:,kk) = Yi;
    
    end
    
    save(fullfile(depth_dataset_path,depths_subdirs{ii},'eeg_topos.mat'), 'eeg_topos', '-v7.3')
    save(fullfile(depth_dataset_path,depths_subdirs{ii},'eeg_topos_xi.mat'), 'eeg_Xi', '-v7.3')
    save(fullfile(depth_dataset_path,depths_subdirs{ii},'eeg_topos_yi.mat'), 'eeg_Yi', '-v7.3')
    
     waitbar(ii/numel(depths_subdirs), w_bar, strcat(w_bar_title,sprintf(':%d %%', floor(ii/numel(depths_subdirs)*100))));
end

close(w_bar);

