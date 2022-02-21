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
        idx = 145;
elseif strcmp(ms,'24_2')
        idx = 150;
elseif strcmp(ms,'25')
    idx = 151;
elseif  strcmp(ms,'25_8')
    idx = 152;
end
eeg_s = (EEG_avg.avg(:,idx) - mean(EEG_avg.avg(:,idx)))/std(EEG_avg.avg(:,idx)) ;

import_directory('./inverse_algorithms/')
%% Single Dipole Fit

[dip,best_loc] = SingleDipoleFit(Le, eeg_s);

[val,idx_max] = max(dip);

figure;
subplot(1,2,1)
contourf(Xi,Yi,Zi)
title('EEG topography.');

subplot(1,2,2)
scatter3(loc(:,1),loc(:,2),loc(:,3),100,dip,'.')
hold on
scatter3(loc(idx_max,1),loc(idx_max,2),loc(idx_max,3),1,dip(idx_max),'y', 'linewidth',5)
title('Dipole fitting localization');
view([291.3 9.2]);
colorbar;
set(gcf,'Position',[60 180 1600 500])

%% sLORETA

b = eeg_s;
alpha = 25;
[u_sLORETA,s] = sLORETA_dir(b,Le,alpha);
figure;
subplot(1,2,1)
contourf(Xi,Yi,Zi)
title('EEG topography.');

subplot(1,2,2)
scatter3(loc(:,1),loc(:,2),loc(:,3),100,u_sLORETA,'.')
title('sLORETA localization');
view([-85.9 11.6]);
colorbar;
set(gcf,'Position',[60 180 1600 500])

