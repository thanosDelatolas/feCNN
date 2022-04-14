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

snr_db = -10:5:20;

distances_cnn = [];
distances_s_loreta = [];
distances_dipole_scan = [];

cnn_dist_db = zeros(size(snr_db));
s_loreta_dist_db = zeros(size(snr_db));
dipole_scan_dist_db = zeros(size(snr_db));

for ii=1:length(snr_db)
    snr = int2str(snr_db(ii));
    fprintf('Load data for evaluation ...\n');
    % read the eeg data
    eeg_signals = double(readNPY(sprintf('./../eval_sim_data/single_source/%sdb/eeg.npy',snr)));
    % prediction created with python and tensorflow
    cnn_predictions = double(readNPY(sprintf('./../eval_sim_data/single_source/%sdb/predicted_sources.npy',snr)));
    % ground truth
    source_centers = double(readNPY(sprintf('./../eval_sim_data/single_source/%sdb/source_centers.npy',snr)));
    
    n_samples = size(cnn_predictions,2);
    
    distances_cnn = zeros(n_samples,1);
    distances_s_loreta = zeros(n_samples,1);
    distances_dipole_scan = zeros(n_samples,1);
    
    fprintf('Evaluate cnn for snr=%sdB\n',snr);
    
    w_bar = waitbar(0, sprintf('Evaluate cnn for snr=%s dB',snr));
   
    for jj=1:n_samples
        
        % noisy eeg signal for sample jj
        eeg_s = eeg_signals(:,jj);
        % ground truth for sample jj
        source =  source_centers(jj)+1;
        
        % prediction of th cnn
        cnn_pred = cnn_predictions(:,jj);
        
        % dipole scanning
        [dipole_scan_out,best_loc] = SingleDipoleFit(Le, eeg_s);
        
        [dipole_scan_out,location_dipole_scan] = create_source_activation_vector(...
            dipole_scan_out,'dipole_fit',cd_matrix);
        
        % sLORETA
        s_loreta_out = sLORETA_with_ori(eeg_s,Le,25);

        [s_loreta_out,location_sloreta] = create_source_activation_vector(...
            s_loreta_out,'sLORETA',cd_matrix);
        
       distances_cnn(jj) = distance_3d_space(loc(source,:), cnn_pred);
       distances_s_loreta(jj) = distance_3d_space(loc(source,:), location_sloreta);
       distances_dipole_scan(jj) = distance_3d_space(loc(source,:),location_dipole_scan);
       
        waitbar(jj/n_samples, w_bar, sprintf('Evaluate cnn for snr %s dB: %d %%',snr,floor(jj/n_samples*100)));
    end
    cnn_dist_db(ii) = mean(distances_cnn);
    s_loreta_dist_db(ii) = mean(distances_s_loreta);
    dipole_scan_dist_db(ii) = mean(distances_dipole_scan);
    close(w_bar);
end
 save('./eval_results/cnn_dist_db.mat','cnn_dist_db')
 save('./eval_results/s_loreta_dist_db.mat','s_loreta_dist_db')
 save('./eval_results/dipole_scan_dist_db.mat','dipole_scan_dist_db')
 
%% Plot the results

load('./eval_results/s_loreta_dist_db.mat')
load('./eval_results/dipole_scan_dist_db.mat')
load('./eval_results/cnn_dist_db.mat')

snr = -10:5:20;

figure;
plot(snr,cnn_dist_db,'linewidth',4);
hold on;
plot(snr,dipole_scan_dist_db,'linewidth',4);
hold on;
plot(snr,s_loreta_dist_db,'linewidth',4);
hold off;
grid on;
legend({'CNN','Dipole Scan', 'sLORETA'},'FontSize',12,'fontweight','bold');
set(gcf,'Position',[220 300 1200 500]);
xlabel('SNR [dB]','fontsize',14,'fontweight','bold');
ylabel('Localization Error [mm]','fontsize',14,'fontweight','bold');
