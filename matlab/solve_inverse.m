clear; close all; clc;

% this script solves the inverse problem using classic methods
% (sLORETA,eLORETA,MNE)

% load leadfield
Le = double(readNPY('../duneuropy/DataOut/leadfield_downsampled_10k.npy'))';

% load source space
load('../duneuropy/Data/dipoles.mat')
load('../duneuropy/Data/dipoles_downsampled_10k.mat')

% load the real data
load('../real_data/EEG_avg.mat')

import_fieldtrip();

lead = [];
lead.pos = source_grid;
lead.unit='mm';
lead.inside= true(size(source_grid,1),1);
count=1;

Leg = Le;
for jj=1:3:(size(Leg,2))
    lead.leadfield{count}=Leg(:,jj:jj+2);
    count=count+1;
end

lead.label=EEG_avg.label;

tmp_data = EEG_avg;

%create a fake head model
cfg=[];
cfg.method='singlesphere';
head = ft_prepare_headmodel(cfg,EEG_avg.elec);

cfg                    = [];
cfg.method             = 'eloreta';                        %specify minimum norm estimate as method
cfg.latency            = toi;            %latency of interest
cfg.grid.pos           = lead.pos;
cfg.grid.inside        = lead.inside;
cfg.grid.unit          = 'mm';
cfg.grid.leadfield     = lead.leadfield;
cfg.headmodel          = head;
cfg.eloreta.prewhiten  = 'yes';                    %prewhiten data
cfg.eloreta.lambda     = 25;                        %regularization parameter
cfg.eloreta.scalesourcecov  = 'yes';                    %scaling the source covariance matrix
source_ft         = ft_sourceanalysis(cfg,tmp_data);

tmp = (source_ft.avg.pow);
para = []; para.title = ['Source localization ']; para.tt=eye(3);

%plot_inv_on_surf(model,para,tmp,0.8,source_grid,mri_t1,1,0,0);%tmp(:,61)

mri_t1        = ft_read_mri(T1_name);

mri_data_scale     = 60;
mri_data_clipping  = .8;

source_activation_mri(mri_t1,mri_data_scale,tmp,source_grid,...
    mri_data_clipping,EEG_avg.time(inv_ind),'EEG Source Localization with eLORETA');