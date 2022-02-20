clear; close all; clc;


Le = double(readNPY('../duneuropy/DataOut/leadfield.npy'))';
load('../duneuropy/Data/dipoles.mat')


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

elec_mm = ft_convert_units(EEG_avg.elec,'mm');

%create a fake head model
cfg=[];
cfg.method='singlesphere';
fake_head = ft_prepare_headmodel(cfg,elec_mm);

%% Minimum norm estimate
cfg         = [];
cfg.method  = 'mne';
cfg.latency = 0.025;%[0.024 0.026];
cfg.grid    = Le;
cfg.headmodel    = fake_head;
cfg.mne.prewhiten = 'yes';
cfg.mne.lambda    = 3;
cfg.mne.scalesourcecov = 'yes';
minimum_norm  = ft_sourceanalysis(cfg,EEG_avg);

T1_name = '../../../Downloads/T1w_1mm_anon.nii';
mri_t1        = ft_read_mri(T1_name);


cfg            = [];
cfg.parameter  = 'avg.pow';
interpolate  = ft_sourceinterpolate(cfg, minimum_norm , mri_t1);

cfg = [];
cfg.funparameter = 'pow';
cfg.method        = 'ortho';
ft_sourceplot(cfg,interpolate);


%% eLORETA

cfg                    = [];
cfg.method             = 'eloreta';                       
cfg.latency            = [0.024 0.026];            %latency of interest
cfg.grid.pos           = lead.pos;
cfg.grid.inside        = lead.inside;
cfg.grid.unit          = 'mm';
cfg.grid.leadfield     = lead.leadfield;
cfg.headmodel          = fake_head;
cfg.eloreta.prewhiten  = 'yes';                    %prewhiten data
cfg.eloreta.lambda     = 3;                        %regularization parameter
cfg.eloreta.scalesourcecov  = 'yes';                    %scaling the source covariance matrix
source_ft         = ft_sourceanalysis(cfg, EEG_avg);


mri_data_scale     = 60;
mri_data_clipping  = .8;

source_activation_mri(mri_t1,mri_data_scale,source_ft.avg.pow,source_grid,...
    mri_data_clipping,EEG_avg.time(151),'EEG Source Localization with eLORETA');


%% dipole fit
cfg = [];
cfg.numdipoles    =  1;
%cfg.grid          = source_grid;
cfg.headmodel     = fake_head;
cfg.grid        = Le;
cfg.elec          = EEG_avg.elec;
cfg.latency       = 0.025;
cfg.nonlinear       = 'no';

dipfit_fem        = ft_dipolefitting(cfg,EEG_avg);

loc = source_grid;
figure;
scatter3(loc(:,1),loc(:,2),loc(:,3),100,'.')
hold on
act_loc = dipfit_fem.dip.pos(1,:) .* 10^-1;

ft_plot_dipole(act_loc, mean(dipfit_fem.dip.mom(1:3,:),2), 'color', 'b','unit','mm')
colorbar; 

