from forward import solve_forward
from simulation import Simulation


fwd = solve_forward()

sim = Simulation(fwd)
sim.simulate(n_samples=100)


