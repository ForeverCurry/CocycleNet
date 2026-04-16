import numpy as np
import torch
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt 

# --------------------------
# Parameters
# --------------------------
numICs = 50
filename_prefix = 'Pendulum_test'
# filename_prefix = 'Pendulum_phase'

x1range = [-3.1, 3.1]
x2range = [-2, 2]
t_span = np.arange(0, 18.0, 0.01)

max_potential = 0.99

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
# def sample_initial_conditions(x1range, x2range, num_samples, seed=0, max_potential=0.99):
#     np.random.seed(seed)
#     X = []
#     E_min = -1.0                      # 最小势能: -cos(theta)，最低为 -1
#     E_max = max_potential             # 允许的最大能量
    
#     for _ in range(num_samples):
#         # 1. 均匀采样能量 E
#         E = np.random.uniform(E_min, E_max)

#         # 2. 均匀采样角度 theta
#         theta = np.random.uniform(*x1range)

#         # 3. 根据能量方程解 ω
#         val = 2 * (E + np.cos(theta))
#         if val < 0:
#             continue  # 该 theta 不可能达到能量 E，重新采样
#         omega = np.random.choice([-1, 1]) * np.sqrt(val)

#         # 限制 omega 范围
#         if x2range[0] <= omega <= x2range[1]:
#             X.append([theta, omega])
    
#     return np.array(X)
# ------------------------------------------
# Sampling Initial Conditions with energy
# ------------------------------------------
def sample_initial_conditions_energy(x1range, x2range, num_samples, seed, max_potential=0.99):
    np.random.seed(seed)
    X = []
    # 均匀采样能量
    # energies = np.linspace(-1.0, max_potential, num_samples)  
    theta = np.linspace(0, x1range[1], num_samples, endpoint=False)
    for t in theta:
        X.append([t, 0])
    return np.array(X)


# --------------------------
# Generate Dataset
# --------------------------
def PendulumFn(x1range, x2range, num_samples, t_span, seed, max_potential):
    # X0 = sample_initial_conditions(x1range, x2range, num_samples, seed, max_potential)
    X0 = sample_initial_conditions_energy(x1range, x2range, num_samples, seed, max_potential)
    dataset = []
    for theta0, omega0 in X0:
        sol = solve_ivp(pendulum_ode, [t_span[0], t_span[-1]], [theta0, omega0],
                        t_eval=t_span, rtol=1e-10, atol=1e-12)
        traj = sol.y.T  # shape: (len(t_span), 2)
        dataset.append(traj)
    return np.array(dataset)  # shape: (num_samples, len(t_span), 2)

# --------------------------
# Save Dataset to CSV
# --------------------------
def save_dataset(filename, data):
    np.save(filename, data)

# --------------------------
# Generate Dataset Sets
# --------------------------
seed = 0
X_train = PendulumFn(x1range, x2range, numICs, t_span, seed=seed, max_potential=max_potential)
save_dataset(f"./dataset/Pendulum/{filename_prefix}.npy", X_train)
dataset = np.load(f"./dataset/Pendulum/{filename_prefix}.npy")
for data in dataset:
    plt.plot(data[:, 0], data[:, 1])
plt.xlabel('Theta')
plt.ylabel('Omega')
plt.title('Pendulum Phase Space')
plt.grid()
plt.savefig(f"{filename_prefix}_phase_space.png", dpi=300)
# plt.show()
