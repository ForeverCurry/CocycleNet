import numpy as np
from scipy.integrate import solve_ivp
import os

# --------------------------
# Parameters
# --------------------------
numICs = 20000
filename_prefix = 'Lorenz'
t_span = np.arange(0, 4.01, 0.01)  # 时间跨度更长些，因为系统可能更混沌

xrange = [-18, 18]
yrange = [-20, 20]
zrange = [0, 50]

# Lorenz parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# 热启动时间
burn_in_time = 10.0       # 热启动时长
burn_in_points = 1000     # 热启动内部积分点数（不存储）
# --------------------------
# Lorenz Dynamics Function
# --------------------------
def lorenz_ode(t, state):
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
        # Step 1: 热启动积分
        sol_burn = solve_ivp(
            lorenz_ode,
            [0, burn_in_time],
            X0[i],
            t_eval=np.linspace(0, burn_in_time, burn_in_points),
            rtol=1e-10, atol=1e-12
        )
        # 取热启动末尾状态
        X_burn_end = sol_burn.y[:, -1]

        # Step 2: 正式积分并存储
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
def generate_sparse_lorenz_trajectory(n_points=10000, dt=0.01, skip_steps=1, 
                                    sigma=10.0, rho=28.0, beta=8.0/3.0,
                                    initial_condition=None):
    """
    生成稀疏的Lorenz轨迹
    
    参数:
    n_points: 最终保留的轨迹点数
    dt: 积分步长
    skip_steps: 每skip_steps个点保留一个点
    sigma, rho, beta: Lorenz系统参数
    initial_condition: 初始条件，默认为[1.0, 1.0, 1.0]
    """
    
    # Lorenz系统的微分方程
    def lorenz(t, state):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    
    # 设置初始条件
    if initial_condition is None:
        initial_condition = [10, 8.5, 30]
    
    # 计算总积分步数，确保有足够的点来稀疏采样
    total_steps = n_points * skip_steps
    t_span = (0, total_steps * dt)
    t_eval = np.linspace(0, total_steps * dt, total_steps)
    
    # 求解微分方程
    sol = solve_ivp(lorenz, t_span, initial_condition, t_eval=t_eval, method='RK45')
    
    # 稀疏采样
    sparse_indices = np.arange(0, total_steps, skip_steps)
    sparse_trajectory = sol.y[:, sparse_indices].T
    
    return np.expand_dims(sparse_trajectory, axis=0)  
# --------------------------
# Save Dataset to NPY
# --------------------------
def save_dataset(filename, data):
    np.save(filename, data)

# --------------------------
# Generate & Save
# --------------------------
seed = 42
# X_lorenz = LorenzFn(xrange, yrange, zrange, numICs, t_span, seed)
vis_lorenz = generate_sparse_lorenz_trajectory()
# save_dataset(f"dataset/Lorenz/{filename_prefix}.npy", X_lorenz)
save_dataset(f"dataset/Lorenz/{filename_prefix}_phase.npy", vis_lorenz)
#  --------------------------------------------------------------------
import matplotlib.pyplot as plt
dataset = np.load(f"dataset/Lorenz/{filename_prefix}_phase.npy")
# data_phase = dataset.reshape(-1,3)
# np.save(f"dataset/Lorenz/{filename_prefix}_phase.npy", data_phase)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 遍历每条轨迹
for data in dataset:
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.plot(x, y, z, alpha=0.6)

# 设置坐标轴标签
ax.set_title('Lorenz Phase Space (3D)')
plt.tight_layout()
plt.savefig(f"{filename_prefix}_phase_space_3d.png")
plt.show()
