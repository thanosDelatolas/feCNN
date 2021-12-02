from forward import solve_forward
from simulation import Simulation
import numpy as np


fwd = solve_forward()

# load eeg and source data from previous simulations
eeg_data = np.load('/media/thanos/Big Data/Thanos/TUC/Thesis/sim_data/eeg_big_sim_2.npy')
source_data = np.load('/media/thanos/Big Data/Thanos/TUC/Thesis/sim_data/sources_big_sim_2.npy')
sim = Simulation(fwd=fwd,source_data=source_data,eeg_data=eeg_data)

from net import EEGNet

eegnet = EEGNet(fwd=fwd,sim=sim)


eegnet.build_model()

eegnet.fit()


