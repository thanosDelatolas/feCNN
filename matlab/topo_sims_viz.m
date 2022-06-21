clear; close all; clc;
load('../real_data/A0206/EEG_avg.mat');

load('../duneuropy/Data/dipoles.mat')
loc = cd_matrix(:,1:3);

[sensors,sensor_labels] = read_elc('./../duneuropy/Data/electrodes.elc');
sensor_labels = split(sensor_labels{4});
sensor_labels = sensor_labels(1:end-1);


layout = '/home/thanos/fieldtrip/template/layout/EEG1010.lay';
[sensors_1010, lay] = compatible_elec(sensor_labels, layout);


sources = double(readNPY('./topo_sims_viz/sources.npy'));
eeg = double(readNPY('./topo_sims_viz/eeg.npy'));


pics = size(eeg,2);
pol = -1;     % correct polarity
scale = 10^6; % scale for eeg data micro volts

for jj=1:pics

    for ii=1:size(EEG_avg.avg,2)
        EEG_avg.avg(:,ii) = eeg(:,jj);
    end
    cfg            = [];
    cfg.zlim       = 'maxmin';
    cfg.comment    = 'no';
    %cfg.commentpos = 'title';
    cfg.xlim       = [0.0141 0.0150];%[0.0245 0.0455];%[EEG_toi_mean_trial EEG_toi_mean_trial+0.01*EEG_toi_mean_trial];
    cfg.layout     = 'EEG1010.lay';
    cfg.fontsize   = 14;

    figure;
    ft_topoplotER(cfg, EEG_avg);
    figure;
    scatter3(loc(:,1),loc(:,2),loc(:,3),100,sources(:,jj),'.')
    view([-103.9 -6.8])
    fprintf('Press enter to continue');
    pause
end


