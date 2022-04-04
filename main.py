from forward import solve_forward
from simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# to load large .mat files
import mat73
from net import EEG_CNN


eeg_topos = mat73.loadmat('./sim_data/eeg_4TeD_topos.mat')['eeg_topos']
eeg_topos = eeg_topos.transpose(2, 0, 1)

sources = np.load('./sim_data/sources_4TeD.npy')
eeg = np.load('./sim_data/eeg_4TeD.npy')
fwd = solve_forward(num_dipoles='25k')


sim = Simulation(fwd=fwd, source_data=sources, eeg_data=eeg)
eeg_cnn = EEG_CNN(sim=sim, eeg_topographies=eeg_topos)

eeg_cnn.build_model()
eeg_cnn.fit()
eeg_cnn.save_nn('./eeg_cnn.h5')