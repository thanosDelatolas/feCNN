from forward import solve_forward
from simulation import Simulation


fwd = solve_forward()

sim = Simulation(fwd, settings={'duration_of_trial' : 0})
sim.simulate(n_samples=100)


import numpy as np

np.save('eeg-test3',sim.eeg_data)