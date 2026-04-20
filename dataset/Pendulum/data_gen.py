import numpy as np
from scipy.integrate import solve_ivp
# --------------------------
# Pendulum Dynamics Function
# --------------------------
def pendulum_ode(t, y):
    theta, omega = y
    return [omega, -np.sin(theta)]

# --------------------------
# Sampling Initial Conditions
# --------------------------
def sample_initial_conditions(x1range, x2range, num_samples, seed, max_potential=0.99):
    np.random.seed(seed)
    X = []
    count = 0
    while len(X) < num_samples:
        theta = np.random.uniform(*x1range)
        omega = np.random.uniform(*x2range)
        potential_energy = 0.5 * omega**2 - np.cos(theta)  # U = 1 - cos(theta)
        if potential_energy <= max_potential:
            X.append([theta, omega])
    return np.array(X)

# --------------------------
# Generate Dataset
# --------------------------
def PendulumFn(x1range, x2range, num_samples, t_span, seed, max_potential):
    X0 = sample_initial_conditions(x1range, x2range, num_samples, seed, max_potential)
    dataset = []
    for theta0, omega0 in X0:
        sol = solve_ivp(pendulum_ode, [t_span[0], t_span[-1]], [theta0, omega0],
                        t_eval=t_span, rtol=1e-10, atol=1e-12)
        traj = sol.y.T  # shape: (len(t_span), 2)
        dataset.append(traj)
    return np.array(dataset)  # shape: (num_samples, len(t_span), 2)

# --------------------------
# Generate Dataset Sets
# --------------------------
if __name__ == "__main__":
    # --------------------------
    # Parameters
    # --------------------------
    numICs = 20000
    filename_prefix = 'Pendulum'

    x1range = [-3.1, 3.1]
    x2range = [-2, 2]
    t_span = np.arange(0, 18.0, 0.01)

    max_potential = 0.99
    seed = 0
    # --------------------------
    X_train = PendulumFn(x1range, x2range, numICs, t_span, seed=seed, max_potential=max_potential)
    np.save(f"./dataset/Pendulum/{filename_prefix}.npy", X_train)