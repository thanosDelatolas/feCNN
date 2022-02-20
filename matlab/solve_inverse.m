clear; close all; clc;


Le = double(readNPY('../duneuropy/DataOut/leadfield.npy'))';
load('../duneuropy/Data/dipoles.mat')
% load the real data
load('../real_data/EEG_avg.mat')
import_directory('./inverse_algorithms/')


%% Single Dipole Fit


eeg_s = (EEG_avg.avg(:,151) - mean(EEG_avg.avg(:,151)))/std(EEG_avg.avg(:,151)) ;

[dip,best_loc] = SingleDipoleFit(Le, eeg_s);


