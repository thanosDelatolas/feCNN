from forward import solve_forward
from simulation import Simulation

from net import EEGNet


fwd = solve_forward()

sim = Simulation(fwd, settings={'duration_of_trial' : 0.2})
sim.simulate(n_samples=100)


# net = EEGNet(fwd=fwd,sim=sim)