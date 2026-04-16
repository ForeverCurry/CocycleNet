import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

# --------------------------
# Lorenz63 参数设置
# --------------------------
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
dt = 0.01  # 步长需与原数据一致

# --------------------------
# 向量化 Lorenz63 微分方程
# --------------------------
def lorenz63_vectorized(state, sigma, rho, beta):
    """
    state 形状: (num_samples, 3)
    返回 形状: (num_samples, 3)
    """
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    # 重新堆叠回 (num_samples, 3)
    return np.stack([dx, dy, dz], axis=1)

# 批量 RK4 步进函数
def rk4_step_batch(state, dt, sigma, rho, beta):
    k1 = lorenz63_vectorized(state, sigma, rho, beta)
    k2 = lorenz63_vectorized(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz63_vectorized(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz63_vectorized(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --------------------------
# 执行扩充逻辑
# --------------------------
def extend_lorenz63_data(original_data, extra_steps):
    """
    original_data: (num_samples, time_steps, 3)
    extra_steps: 需要额外积分的步数
    """
    num_samples, _, _ = original_data.shape
    
    # 获取每个轨迹的最后一个状态作为起点
    current_state = original_data[:, -1, :].copy()
    
    # 预分配扩展部分的内存
    extra_segments = np.zeros((num_samples, extra_steps, 3))
    
    print(f"正在扩充 {num_samples} 条 Lorenz63 轨迹...")
    for s in trange(extra_steps):
        current_state = rk4_step_batch(current_state, dt, sigma, rho, beta)
        extra_segments[:, s, :] = current_state
        
    # 拼接原始数据与新生成的数据
    extended_data = np.concatenate([original_data, extra_segments], axis=1)
    return extended_data

# --------------------------
# 示例运行与可视化
# --------------------------
if __name__ == "__main__":
    # 模拟已有数据: 10个样本，每个500步
    data_path = './dataset/Lorenz/Lorenz.npy'
    data = np.load(data_path)
    
    # 扩充 1000 步
    new_data = extend_lorenz63_data(data, extra_steps=1000)
    output_path = './dataset/Lorenz/Lorenz_extend.npy'
    np.save(output_path, new_data)
    # 可视化其中一个轨迹的 3D 吸引子
    sample_to_plot = new_data[0]
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始部分（蓝色）
    ax.plot(sample_to_plot[:401, 0], sample_to_plot[:401, 1], sample_to_plot[:401, 2], 
            'b', label='Original', alpha=0.6)
    # 绘制扩充部分（红色）
    ax.plot(sample_to_plot[500:, 0], sample_to_plot[500:, 1], sample_to_plot[500:, 2], 
            'r', label='Extended', alpha=0.8)
    
    ax.set_title("Lorenz63 Data Extension")
    ax.legend()
    plt.savefig('Lorenz63 trajectory')