import numpy as np
from scipy.integrate import solve_ivp

# --------------------------
# Lorenz Dynamics Function
# --------------------------
def lorenz_ode(t, state):
    # Lorenz parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# --------------------------
# Sample Initial Conditions
# --------------------------
def sample_initial_conditions(xr, yr, zr, num_samples, seed):
    np.random.seed(seed)
    x = np.random.uniform(*xr, size=(num_samples, 1))
    y = np.random.uniform(*yr, size=(num_samples, 1))
    z = np.random.uniform(*zr, size=(num_samples, 1))
    return np.hstack([x, y, z])

# --------------------------
# Generate Dataset
# --------------------------
def LorenzFn(xr, yr, zr, num_samples, t_span, seed):
    X0 = sample_initial_conditions(xr, yr, zr, num_samples, seed)
    dataset = []
    for i in range(num_samples):
        # warm up integration to reach the attractor
        sol_burn = solve_ivp(
            lorenz_ode,
            [0, burn_in_time],
            X0[i],
            t_eval=np.linspace(0, burn_in_time, burn_in_points),
            rtol=1e-10, atol=1e-12
        )
        # select the last point as the new initial condition for the main integration
        X_burn_end = sol_burn.y[:, -1]

        # integrate the main trajectory
        sol = solve_ivp(
            lorenz_ode,
            [t_span[0], t_span[-1]],
            X_burn_end,
            t_eval=t_span,
            rtol=1e-10, atol=1e-12
        )
        traj = sol.y.T  # shape: (len(t_span), 3)
        dataset.append(traj)
    return np.array(dataset)  # shape: (num_samples, len(t_span), 3)
# --------------------------
# Generate & Save
# --------------------------
if __name__ == "__main__":
    # --------------------------
    # Parameters
    # --------------------------
    numICs = 20000
    filename_prefix = 'Lorenz'
    t_span = np.arange(0, 10.01, 0.01)  

    xrange = [-18, 18]
    yrange = [-20, 20]
    zrange = [0, 50]

    # warm-up parameters
    burn_in_time = 10.0       
    burn_in_points = 1000  
    seed = 42
    # --------------------------
    X_lorenz = LorenzFn(xrange, yrange, zrange, numICs, t_span, seed)
    np.save(f"dataset/Lorenz/{filename_prefix}.npy", X_lorenz)