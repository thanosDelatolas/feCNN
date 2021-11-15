from forward import solve_forward
from simulation import Simulation


fwd = solve_forward()
print(fwd.leadfield.shape)

simulation = Simulation(fwd)

sim = simulation.simulate(n_samples=100)

print(sim.shape)

