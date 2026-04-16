import numpy as np
from numpy.fft import fft, ifft
import time
np.random.seed(2025)
# 参数设置 (调整为文献标准)
L = 8.0 * np.pi       # 域长度
N = 128               # 傅里叶模式数
M = 128               # 空间离散点数
dt = 0.1              # 减小时间步长以提高稳定性

spin_up_steps = 1000   # 预热步数 (达到统计稳态)
trajectory_length = 100  # 记录步数
batch_size = 20000  
initial_scale = 1   # 增大初始扰动幅值
T = trajectory_length * dt
# 波数向量
k = 2 * np.pi * np.fft.fftfreq(M, d=L/M)

# 线性算子 + 正则化
linear_operator = k**2 - k**4 + 1e-12 * np.ones_like(k)

# 去混掩码 (2/3规则)
k_max = (2/3) * (M/2)  # 最大未混叠波数
dealias_mask = np.abs(k) <= k_max

# 生成初始条件函数 (增大幅值)
def generate_initial_condition(N, initial_scale=10.0, modes=range(1,N//4)):
    """
    返回频域的初始傅里叶系数 v0 (长度 N, complex128),
    激发指定 modes（正模），并在对应负模处写入共轭对称以保证实值解。
    """
    v0 = np.zeros(N, dtype=np.complex128)
    for m in modes:
        if m >= N//2:
            continue
        val = initial_scale * (np.random.randn() + 1j * np.random.randn())
        v0[m] = val
        v0[-m] = np.conj(val)   # 保证实数信号
    return v0

# 带去混叠的非线性项计算
def compute_nonlinear(u):
    u_hat = fft(u)
    u_sq = np.real(ifft(u_hat))**2
    N_hat = fft(u_sq)
    N_hat[~dealias_mask] = 0  # 应用去混叠
    return -0.5j * k * N_hat

# 时间步进函数
def time_step(v_prev, v_curr, N_prev_hat):
    # 计算当前非线性项 (带去混叠)
    u_curr = np.real(ifft(v_curr))
    N_curr_hat = compute_nonlinear(u_curr)
    
    # 二阶半隐式格式
    numerator = (1 + 0.5*dt*linear_operator) * v_curr + \
                dt * (1.5*N_curr_hat - 0.5*N_prev_hat)
    denominator = 1 - 0.5*dt*linear_operator
    
    # 避免除零
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    v_next = numerator / denominator
    
    return v_next, N_curr_hat

# 生成批量数据 (增加预热阶段)
def generate_batch_data():
    data = np.zeros((batch_size, trajectory_length, M), dtype=np.float32)
    
    for b in range(batch_size):
        if b % 100 == 0:
            print(f"生成轨迹 {b+1}/{batch_size}")
        
        # 生成初始条件
        v0 = generate_initial_condition()
        u0 = np.real(ifft(v0))
        
        # === 预热阶段 (达到统计稳态) ===
        v_prev = v0
        # 第一步
        N0_hat = compute_nonlinear(u0)
        v_curr = (v0 + dt * N0_hat) / (1 - dt * linear_operator)
        u_prev = u0
        
        # 预热步进
        for _ in range(spin_up_steps):
            u_curr = np.real(ifft(v_curr))
            v_next, N_curr_hat = time_step(v_prev, v_curr, N0_hat)
            # 更新变量
            v_prev = v_curr
            v_curr = v_next
            N0_hat = N_curr_hat
        
        # === 记录阶段 ===
        trajectory = np.zeros((trajectory_length, M), dtype=np.float32)
        trajectory[0] = np.real(ifft(v_curr))
        
        for t in range(1, trajectory_length):
            u_curr = np.real(ifft(v_curr))
            v_next, N_curr_hat = time_step(v_prev, v_curr, N0_hat)
            trajectory[t] = np.real(ifft(v_next))
            
            # 更新变量
            v_prev = v_curr
            v_curr = v_next
            N0_hat = N_curr_hat
        
        data[b] = trajectory
    
    return data

import numpy as np
from joblib import Parallel, delayed

def ks_etdrk4(L=14*np.pi, N=256, dt=0.1, spin_up_steps =1000, nsteps=500, seed=0, dealias=True):
    # np.random.seed(seed)
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)

    # 正确的角波数（实数）
    k = 2*np.pi * np.fft.fftfreq(N, d=dx)    # real angular wavenumbers

    # 线性算子（谱空间）
    Lk = k**2 - k**4   # 注意：这是 k^2 - k^4

    # ETDRK4 预计算（Trefethen 配方）
    E  = np.exp(dt * Lk)
    E2 = np.exp(dt * Lk / 2.0)
    M  = 32
    r  = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = dt * Lk[:, None] + r[None, :]
    Q  = dt * np.mean((np.exp(LR/2.0) - 1.0) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
    f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)
        # dealias mask (2/3 rule)
    if dealias:
        cutoff = N//3   # 2/3 rule => keep modes |k| <= N/3
        freqs = np.fft.fftfreq(N) * N   # indices -N/2..N/2-1
        mask = (np.abs(freqs) <= cutoff).astype(np.float64)
    else:
        mask = np.ones(N, dtype=np.float64)

    # nonlinear term: compute -0.5j * k * FFT(u^2) with dealias applied to FFT(u^2)
    def nonlinear_hat_from_u(u):
        u2 = u**2
        U2_hat = np.fft.fft(u2)
        if dealias:
            U2_hat = U2_hat * mask
        return -0.5j * k * U2_hat   # complex array

    # ---- initial condition: get frequency-domain v0 ----
    v0 = generate_initial_condition(N, initial_scale=1.0, modes=(1,2,3))
    u_hat = v0.copy()   # u_hat is spectral coefficients (complex)

    # optional: small check print
    # print("init max abs u:", np.max(np.abs(np.real(np.fft.ifft(u_hat)))))

    # ---- spin-up ----
    for _ in range(spin_up_steps):
        u = np.real(np.fft.ifft(u_hat))
        N_hat = nonlinear_hat_from_u(u)

        a_hat = E2 * u_hat + Q * N_hat
        Na_hat = nonlinear_hat_from_u(np.real(np.fft.ifft(a_hat)))

        b_hat = E2 * u_hat + Q * Na_hat
        Nb_hat = nonlinear_hat_from_u(np.real(np.fft.ifft(b_hat)))

        c_hat = E2 * a_hat + Q * (2*Nb_hat - N_hat)
        Nc_hat = nonlinear_hat_from_u(np.real(np.fft.ifft(c_hat)))

        u_hat = E * u_hat + N_hat * f1 + 2*(Na_hat + Nb_hat) * f2 + Nc_hat * f3

        # stability check (optional)
        if np.max(np.abs(np.real(np.fft.ifft(u_hat)))) > 1e6:
            raise RuntimeError("solution blew up during spin-up; reduce dt or initial_scale")

    # ---- record data ----
    data = np.zeros((nsteps, N), dtype=np.float64)
    for n in range(nsteps):
        u = np.real(np.fft.ifft(u_hat))
        data[n,:] = u

        N_hat = nonlinear_hat_from_u(u)

        a_hat = E2 * u_hat + Q * N_hat
        Na_hat = nonlinear_hat_from_u(np.real(np.fft.ifft(a_hat)))

        b_hat = E2 * u_hat + Q * Na_hat
        Nb_hat = nonlinear_hat_from_u(np.real(np.fft.ifft(b_hat)))

        c_hat = E2 * a_hat + Q * (2*Nb_hat - N_hat)
        Nc_hat = nonlinear_hat_from_u(np.real(np.fft.ifft(c_hat)))

        u_hat = E * u_hat + N_hat * f1 + 2*(Na_hat + Nb_hat) * f2 + Nc_hat * f3

    return data

def is_chaotic(u, threshold_std=0.5, threshold_entropy=0.5):
    """
    通过空间标准差和谱熵检测 KS 方程是否进入混沌态。
    
    参数:
    u: 形状为 (N,) 的空间场数据 (实空间)
    threshold_std: 空间波动的标准差阈值。混沌态通常有较强的振幅波动。
    threshold_entropy: 谱熵阈值。越接近混沌，能量分布越分散，熵值越高。
    """
    # 1. 检查振幅波动强度 (混沌态振幅通常在 -2 到 2 之间剧烈跳变)
    std_val = np.std(u)
    
    # 2. 计算谱熵 (Spectral Entropy)
    # 能量分布越广，代表时空结构越复杂
    u_hat = np.fft.fft(u)
    psd = np.abs(u_hat[:len(u)//2])**2  # 取正频段功率谱
    psd_norm = psd / np.sum(psd)        # 归一化
    
    # 避免 log(0)
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
    # 逻辑判断：振幅够大且频谱够乱
    is_fine = (std_val > threshold_std) and (spectral_entropy > threshold_entropy)
    
    return is_fine, std_val, spectral_entropy

def safe_ks_etdrk4( **kwargs):
    """封装：检测 NaN，如果有则重试"""
    max_retries = 50
    for attempt in range(max_retries):
        data = ks_etdrk4(**kwargs)
        if not np.isnan(data).any():
            # 检查轨迹的最后一步是否达到了混沌态
            passed, s_val, e_val = is_chaotic(data[-1, :])
            
            if passed:
                return data
            else:
                print(f"检测到非混沌轨迹，已舍弃。尝试次数: {attempt}", end='\r')
                
        else:
            print(f"[警告] 出现 NaN，重试 {attempt+1}/{max_retries}...")
    raise RuntimeError(f"多次重试仍然产生 NaN！")

def generate_dataset(num_init=10000, T=100, N=256, L=14*np.pi, dt=0.1, spin_up_steps=20, n_jobs=-1):
    """并行生成 KS 数据集: (num_init, T, N)"""
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(safe_ks_etdrk4)(L=L, N=N, dt=dt, spin_up_steps=spin_up_steps, nsteps=T)
        for _ in range(num_init)
    )
    return np.array(results)

def compute_energy(u, L):
    """
    计算能量 E(t) = (1/L) * ∫ u(x,t)^2 dx
    u: array (N,), 空间解
    L: 空间长度
    """
    N = len(u)
    dx = L / N
    return np.sum(u**2) * dx / L

# 示例：生成一个小数据集
if __name__ == "__main__":
    L= 8*np.pi
    dt = 0.25
    spin_up_steps = 1000   # 预热步数 (达到统计稳态)
    trajectory_length = 400  # 记录步数
    batch_size = 20000  
    T = trajectory_length * dt
    # ks_data = generate_dataset(num_init=20000, T=trajectory_length, N=128, L=L, dt=dt, n_jobs=-1)
    # print("数据形状:", ks_data.shape)  # (20, 100, 256)
    # np.save(f'./dataset/KS/KS_8pi_0.25.npy', ks_data[:,100:,])
# # 生成数据
# print(f"开始生成 {batch_size} 条轨迹，预热 {spin_up_steps} 步，记录 {trajectory_length} 步...")
# start_time = time.time()
# ks_data = generate_batch_data()
# elapsed_time = time.time() - start_time

# # 分析数据范围
# min_val = np.min(ks_data)
# max_val = np.max(ks_data)
# mean_val = np.mean(ks_data)
# std_val = np.std(ks_data)

# print(f"数据范围: [{min_val:.2f}, {max_val:.2f}]")
# print(f"平均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
# print(f"生成时间: {elapsed_time:.2f} 秒")

# # 保存数据
# np.save('./dataset/KS/KS.npy', ks_data)
# print("数据已保存为 KS.npy")

    data = np.load(f"./dataset/KS/KS_8pi_0.25.npy")
    print(data.shape)
    print(np.isnan(data).any())      # 是否存在 NaN
    print(np.isinf(data).any())      # 是否存在 Inf
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MultipleLocator
    batch_idx = 5


    solution=data[batch_idx]
    # print(is_chaotic(data[batch_idx,-1, :]))
    nsteps = solution.shape[0]
    energies = np.array([compute_energy(data[n], L) for n in range(nsteps)])
    print(np.min(solution), np.max(solution))
    # 获取时间和空间维度
    t_steps, x_points = solution.shape
    # 创建网格
    time = np.arange(t_steps)  # 时间点 [0, 1, 2, ..., t_steps-1]
    space = np.linspace(0, L, x_points)  # 空间位置 [0, 8π]

    # T, X = np.meshgrid(time, space, indexing='ij')  # 创建网格，索引为[i,j]对应(time[i], space[j])
    fig = plt.figure(figsize=(12, 12))
        
    # 创建伪彩色图
    norm = Normalize(vmin=np.min(solution), vmax=np.max(solution))
    ax1 = fig.add_subplot(2, 1, 1)
    im1 = ax1.imshow( solution.T,  extent=[0, T, 0, L], aspect='auto', origin='lower', cmap='RdBu_r')

    # 添加颜色条
    cbar = fig.colorbar(im1, label='u(x,t)')
    cbar.ax.tick_params(labelsize=12)

    # 设置轴标签和标题
    ax1.set_xlabel('Time (t)', fontsize=14, labelpad=10)
    ax1.set_ylabel('Space (x)', fontsize=14, labelpad=10)
    ax1.set_title(f'Kuramoto-Sivashinsky Equation Solution (Batch {batch_idx})', fontsize=16, pad=20)

    # # 设置刻度
    # ax1.set_xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    # 设置空间轴刻度为π的倍数
    yticks = np.arange(0, L+0.5*np.pi, np.pi)
    ax1.set_yticks(yticks, [f'{int(x/np.pi)}π' if x > 0 else '0' for x in yticks])

    # # 设置时间轴主刻度和次刻度
    # plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    # plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(np.arange(0+1,T+1), energies, lw=1.5)
    # ax2.set_xlabel("Time")
    # ax2.set_ylabel("Energy E(t)")
    # ax2.set_title("Energy evolution of KS system")
    # ax2.grid(True)
    # # 添加网格线
    # plt.grid(True, linestyle='--', alpha=0.3, which='both')

    # 调整布局
    plt.tight_layout()
    plt.savefig('KS.png', dpi=200, bbox_inches='tight')