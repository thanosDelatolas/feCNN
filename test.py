from forward import solve_forward
from simulation import Simulation


fwd = solve_forward()

simulation = Simulation(fwd)

sim = simulation.simulate(n_samples=100)


