clear;
close all;
clc;

import_fieldtrip();

data_name              = '/media/thanos/Elements/thanos/sep_sef/sep_sef.ds';      

% Read events
cfg                    = [];
cfg.trialdef.prestim   = 0.1;                   % in seconds
cfg.trialdef.poststim  = 0.2;                   % in seconds
cfg.trialdef.eventtype = 'rightArm';            % get a list of the available types
cfg.dataset            = data_name;             % set the name of the dataset
cfg_tr_def             = ft_definetrial(cfg);   % read the list of the specific stimulus

% segment data according to the trial definition
cfg                    = [];
cfg.dataset            = data_name;
cfg.channel            = 'EEG1010';             % define channel type
data_eeg               = ft_preprocessing(cfg); % read raw data
data_eeg               = ft_redefinetrial(cfg_tr_def, data_eeg);



cfg                = [];
cfg.hpfilter       = 'yes';        % enable high-pass filtering
cfg.lpfilter       = 'yes';        % enable low-pass filtering
cfg.hpfreq         = 20;           % set up the frequency for high-pass filter
cfg.lpfreq         = 250;          % set up the frequency for low-pass filter
cfg.dftfilter      = 'yes';        % enable notch filtering to eliminate power line noise
cfg.dftfreq        = [50 100 150]; % set up the frequencies for notch filtering
cfg.baselinewindow = [-0.1 -0.02];    % define the baseline window
data_eeg           = ft_preprocessing(cfg,data_eeg);


data_eeg = rmfield(data_eeg, 'grad');

cfg        = [];
cfg.metric = 'zvalue';  % use by default zvalue method
cfg.method = 'summary'; % use by default summary method

data_eeg       = ft_rejectvisual(cfg,data_eeg);


cfg                   = [];
cfg.preproc.demean    = 'yes';    % enable demean to remove mean value from each single trial
cfg.covariance        = 'yes';    % calculate covariance matrix of the data
cfg.covariancewindow  = [-0.1 0]; % calculate the covariance matrix for a specific time window
EEG_avg               = ft_timelockanalysis(cfg, data_eeg);


cfg               = [];
cfg.reref         = 'yes';
cfg.refchannel    = 'all';
cfg.refmethod     = 'avg';
EEG_avg           = ft_preprocessing(cfg,EEG_avg);

% save MEG_avg MEG_avg
% save EEG_avg EEG_avg

cfg = [];
cfg.method = 'amplitude';
EEG_gmfp = ft_globalmeanfield(cfg, EEG_avg);

figure;

pol = -1;     % correct polarity
scale = 10^6; % scale for eeg data micro volts

signal_EEG = scale*pol*EEG_avg.avg; % add single trials in a new value

% plot single trial together with global mean field power
h1 = plot(EEG_avg.time,signal_EEG,'color',[0,0,0.5]);
hold on;
h2 = plot(EEG_avg.time,scale*EEG_gmfp.avg,'color',[1,0,0],'linewidth',1);



mx = max(max(signal_EEG));
mn = min(min(signal_EEG));

% select time of interest for the source reconstruction later on
idx = find(EEG_avg.time>0.024 & EEG_avg.time<=0.027);
toi = EEG_avg.time(idx);

[mxx,idxm] = max(max(abs(EEG_avg.avg(:,idx))));
EEG_toi_mean_trial = toi(idxm);


cfg          = [];
cfg.fontsize = 6;
cfg.layout   = 'EEG1010.lay';
cfg.fontsize = 14;
cfg.ylim     = [-5e-6 5e-6];
cfg.xlim     = [-0.1 0.2];

figure;
ft_multiplotER(cfg, EEG_avg);

set(gcf, 'Position',[1 1 1200 800])



cfg            = [];
%cfg.zlim       = 'maxmin';
% cfg.comment    = 'xlim';
% cfg.commentpos = 'title';
cfg.xlim       = [0.0245 0.0455];%[EEG_toi_mean_trial EEG_toi_mean_trial+0.01*EEG_toi_mean_trial];
cfg.layout     = 'EEG1010.lay';
%cfg.fontsize   = 14;

figure;
ft_topoplotER(cfg, EEG_avg);
set(gcf, 'Position',[1 1 1200 800])

