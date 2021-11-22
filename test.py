from forward import solve_forward
from simulation import Simulation


fwd = solve_forward()

sim = Simulation(fwd)
#s = sim.simulate_sources(n_samples=50)
s1 = sim.simulate_source()
print(s1.shape)

