clear; close all; clc;

import_fieldtrip();

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);

depth_dataset_path = '../eval_sim_data/depth/-10db/';
depths_struct = dir(fullfile(depth_dataset_path,'*'));

% list of subfolders of depth_dataset_path.
depths_subdirs = setdiff({depths_struct([depths_struct.isdir]).name},{'.','..'});
for ii = 1:numel(depths_subdirs)
    % file extention or *
    subdir = dir(fullfile(depth_dataset_path,depths_subdirs{ii},'.npy'));
    C = {subdir(~[subdir.isdir]).name}; % files in subfolder.
    for jj = 1:numel(C)
        F = fullfile(depth_dataset_path,depths_subdirs{ii},C{jj})
        
    end
end