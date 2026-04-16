import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.colors import TwoSlopeNorm
import os
import glob
from torch import Tensor
from tqdm import trange
# Configure plotting aesthetics (inspired by SIAM paper style)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",  # use LaTeX-like math fonts
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300,
})

colors = {
    'siam_blue': "#003366",  # deep blue
    'siam_red': "#990000",   # deep red
    'siam_gray': "#444444",  # neutral gray for guides
}

# Default colorbar options used across plotting helpers
CBAR_OPTS = dict(orientation="horizontal", shrink=0.6, aspect=25, pad=0.05)
def visulization_phase_error(x, pre, save_path=None):
    """
    绘制相图与误差图结合的图
    x: Tensor [B, T, D] - 真实数据
    pre: Tensor [B, T, D] - 预测数据
    """
    B, T, D = x.shape
    x_flat = x.reshape(-1, D).cpu().detach().numpy()
    pre_flat = pre.reshape(-1, D).cpu().detach().numpy()

    # 计算误差
    # error_flat = np.linalg.norm(pre_flat - x_flat, axis=1) / np.clip(np.linalg.norm(x_flat, axis=1), a_min=1e-3, a_max=1e10)
    error_flat = np.linalg.norm(pre_flat - x_flat, axis=1) 
    if D <= 3:
        # 维度不大于3，保持之前的散点图形式
        n_rows = 2
        n_cols = 1
        error_max = np.max(error_flat)
        error_min = np.min(error_flat)

        fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
        cax_error = fig.add_axes([0.85, 0.3, 0.02, 0.4])

        ax_phase = fig.add_subplot(n_rows, n_cols, 1, projection='3d' if D == 3 else None)

        if D == 3:
            sc_phase = ax_phase.scatter(
                x_flat[:, 0], x_flat[:, 1], x_flat[:, 2],
                c=error_flat, cmap='viridis', s=4,
                vmin=error_min, vmax=error_max
            )
            ax_phase.set_xlabel("dim 0")
            ax_phase.set_ylabel("dim 1")
            ax_phase.set_zlabel("dim 2")
        else:
            sc_phase = ax_phase.scatter(
                x_flat[:, 0], x_flat[:, 1],
                c=error_flat, cmap='viridis', s=4,
                vmin=error_min, vmax=error_max
            )

        ax_phase.set_title("Phase Plot Colored by Error")
        fig.colorbar(sc_phase, cax=cax_error, label='Prediction Error', extend='both')

        ax_pre = fig.add_subplot(n_rows, n_cols, 2, projection='3d' if D == 3 else None)
        if D == 3:
            ax_pre.scatter(pre_flat[:, 0], pre_flat[:, 1], pre_flat[:, 2], color='blue', s=4)
        else:
            ax_pre.scatter(pre_flat[:, 0], pre_flat[:, 1], color='blue', s=4)
        ax_pre.set_title("Prediction Trajectory")

        plt.tight_layout(rect=[0, 0, 0.8, 1])
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)

    else:
        # D > 3，画热力图
        # 先reshape回 [B, T, D]
        x_np = x.cpu().detach().numpy()
        pre_np = pre.cpu().detach().numpy()
        error_np = np.abs(pre_np - x_np)  # 误差矩阵 shape: [B, T]

        # 这里我们展示第一个batch的数据（你可以修改展示方式）
        x_show = x_np[0].T  # shape (D, T)
        pre_show = pre_np[0].T
        error_show = error_np[0].T  # 误差每个时间点的总误差，shape (1, T)

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

        im0 = axs[0].imshow(x_show, aspect='auto', cmap='jet')
        axs[0].set_title("True Values (State x Time)")
        axs[0].set_ylabel("State Dim")
        fig.colorbar(im0, ax=axs[0], orientation='vertical')

        im1 = axs[1].imshow(pre_show, aspect='auto', cmap='jet')
        axs[1].set_title("Predicted Values (State x Time)")
        axs[1].set_ylabel("State Dim")
        fig.colorbar(im1, ax=axs[1], orientation='vertical')

        im2 = axs[2].imshow(error_show, aspect='auto', cmap='inferno')
        axs[2].set_title("Error plot")
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Error")
        fig.colorbar(im2, ax=axs[2], orientation='vertical')

        if save_path:
            plt.savefig(save_path)
        plt.close(fig)

def visualization_error(t, x, pre1, pre2, pre3, labels=("CocycleNet", "DeepKoopman", "LRAN"), save_path=None):
    """
    三行可视化：
    第1行：真值
    第2行：模型1误差
    第3行：模型2误差

    x    : Tensor [B, T, D] 真实数据
    pre1 : Tensor [B, T, D] 模型1预测
    pre2 : Tensor [B, T, D] 模型2预测
    """

    B, T, D = x.shape

    x_flat = x.reshape(-1, D).cpu().detach().numpy()
    pre1_flat = pre1.reshape(-1, D).cpu().detach().numpy()
    pre2_flat = pre2.reshape(-1, D).cpu().detach().numpy()
    pre3_flat = pre3.reshape(-1, D).cpu().detach().numpy()

    if D <= 3:
        err1_flat = np.linalg.norm(pre1_flat - x_flat, axis=1)
        err2_flat = np.linalg.norm(pre2_flat - x_flat, axis=1)
        err3_flat = np.linalg.norm(pre3_flat - x_flat, axis=1)

        # 统一误差色标，保证可比性
        err_min = min(err1_flat.min(), err2_flat.min())
        err_max = max(err1_flat.max(), err2_flat.max())
        # =========================
        # 相图（D <= 3）
        # =========================
        fig = plt.figure(figsize=(12, 12))
        cax = fig.add_axes([0.25, 0.2, 0.5, 0.015])

        # ---------- 真值 ----------
        ax0 = fig.add_subplot(2, 2, 1, projection='3d' if D == 3 else None)
        if D == 3:
            sc0 = ax0.scatter(x_flat[:, 0], x_flat[:, 1], x_flat[:, 2],
                        s=4, alpha=0.8)
        else:
            sc0 = ax0.scatter(x_flat[:, 0], x_flat[:, 1],
                        s=4, alpha=0.8)

        ax0.set_title("True Trajectory", fontsize=16, fontweight='bold')
        # ---------- 模型1误差 ----------
        ax1 = fig.add_subplot(2, 2, 2, projection='3d' if D == 3 else None)
        if D == 3:
            sc1 = ax1.scatter(
                x_flat[:, 0], x_flat[:, 1], x_flat[:, 2],
                c=err1_flat, cmap='viridis', s=4,
                vmin=err_min, vmax=err_max
            )
        else:
            sc1 = ax1.scatter(
                x_flat[:, 0], x_flat[:, 1],
                c=err1_flat, cmap='viridis', s=4,
                vmin=err_min, vmax=err_max
            )
        ax1.set_title(f"{labels[0]} Error plot", fontsize=16, fontweight='bold')

        # ---------- 模型2误差 ----------
        ax2 = fig.add_subplot(2, 2, 3, projection='3d' if D == 3 else None)
        if D == 3:
            sc2 = ax2.scatter(
                x_flat[:, 0], x_flat[:, 1], x_flat[:, 2],
                c=err2_flat, cmap='viridis', s=4,
                vmin=err_min, vmax=err_max
            )
        else:
            sc2 = ax2.scatter(
                x_flat[:, 0], x_flat[:, 1],
                c=err2_flat, cmap='viridis', s=4,
                vmin=err_min, vmax=err_max
            )

        ax2.set_title(f"{labels[1]} Error plot", fontsize=16, fontweight='bold')

        ax3 = fig.add_subplot(2, 2, 4, projection='3d' if D == 3 else None)
        if D == 3:
            sc3 = ax3.scatter(
                x_flat[:, 0], x_flat[:, 1], x_flat[:, 2],
                c=err3_flat, cmap='viridis', s=4,
                vmin=err_min, vmax=err_max
            )
        else:
            sc3 = ax3.scatter(
                x_flat[:, 0], x_flat[:, 1],
                c=err3_flat, cmap='viridis', s=4,
                vmin=err_min, vmax=err_max
            )

        ax3.set_title(f"{labels[2]} Error plot", fontsize=16, fontweight='bold')
 
        cbar = fig.colorbar(sc3, cax=cax, orientation='horizontal')
        cbar.set_label("Relative MSE", fontsize=14)
        for ax in (ax0,ax1,ax2,ax3):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.2, 1, 1])
        if save_path:
            plt.savefig(save_path, transparent=True, bbox_inches="tight")
        plt.close(fig)

    else:
        # =========================
        # 热力图（D > 3）
        # =========================
        x_np = x.cpu().detach().numpy()
        err1_np = np.abs(pre1.cpu().detach().numpy() - x_np)
        err2_np = np.abs(pre2.cpu().detach().numpy() - x_np)
        err3_np = np.abs(pre3.cpu().detach().numpy() - x_np)
         # 统一误差色标，保证可比性
        err_min = min(err1_np[0].min(), err2_np[0].min(), err3_np[0].min())
        err_max = max(err1_np[0].max(), err2_np[0].max(), err3_np[0].max())
        # =========================
        # 预定义右侧 colorbar 通道
        # =========================
        cbar_left = 0.92
        cbar_width = 0.015
        # 仅展示第一个 batch
        x_show = x_np[0].T          # (D, T)
        err1_show = err1_np[0].T
        err2_show = err2_np[0].T
        err3_show = err3_np[0].T

        fig, axs = plt.subplots(4, 1, figsize=(9, 12))

        im0 = axs[0].imshow(x_show, aspect='auto', cmap='turbo',
                            extent=[t[0], t[-1], 0, err1_show.shape[0]])
        axs[0].set_title(f"True Trajectory", fontsize=18, fontweight='bold')
        axs[0].set_ylabel("State Dim")
        cax0 = fig.add_axes([cbar_left, 0.8, cbar_width, 0.15])
        cbar0 = fig.colorbar(im0, cax=cax0)
        cbar0.ax.tick_params(labelsize=10)

        im1 = axs[1].imshow(err1_show, aspect='auto', cmap='magma',vmin=err_min, vmax=err_max,
                            extent=[t[0], t[-1], 0, err1_show.shape[0]])
        axs[1].set_title(f"{labels[0]} Error plot", fontsize=18, fontweight='bold')
        axs[1].set_ylabel("State Dim")

        im2 = axs[2].imshow(err2_show, aspect='auto', cmap='magma',vmin=err_min, vmax=err_max,
                            extent=[t[0], t[-1], 0, err1_show.shape[0]])
        axs[2].set_title(f"{labels[1]} Error plot", fontsize=18, fontweight='bold')
        axs[2].set_ylabel("State Dim")

        im3 = axs[3].imshow(err3_show, aspect='auto', cmap='magma',vmin=err_min, vmax=err_max,
                            extent=[t[0], t[-1], 0, err1_show.shape[0]])
        axs[3].set_title(f"{labels[2]} Error plot", fontsize=18, fontweight='bold')
        axs[3].set_xlabel("Time (Lyapunov Time)")
        axs[3].set_ylabel("State Dim")
        # =======================
        cax_err = fig.add_axes([cbar_left, 0.18, cbar_width, 0.45])
        cbar_err = fig.colorbar(im3, cax=cax_err)
        cbar_err.ax.tick_params(labelsize=10)
        plt.tight_layout(rect=[0, 0, cbar_left - 0.02, 1])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def visualization_time_series(
    x, x_1, x_2,
    labels=("KoopGen", "DeepKoopman"),
    save_path=None,
    max_dims=None
):
    """
    每个空间维单独绘制时间序列图（真值 + 两个预测叠加）

    x    : Tensor [B, T, D] - 真实数据
    x_1  : Tensor [B, T, D] - 预测结果 1
    x_2  : Tensor [B, T, D] - 预测结果 2
    """

    B, T, D = x.shape

    # 只展示第一个 batch
    x_np   = x[0].cpu().detach().numpy()    # [T, D]
    x1_np  = x_1[0].cpu().detach().numpy()  # [T, D]
    x2_np  = x_2[0].cpu().detach().numpy()  # [T, D]

    time = np.arange(T)

    # 限制最多绘制的维度数
    dims_to_plot = D if max_dims is None else min(D, max_dims)

    for d in range(dims_to_plot):
        fig, ax = plt.subplots(figsize=(10, 4))

        # --- Ground truth ---
        ax.plot(
            time, x_np[:, d],
            color='black',
            linewidth=2.0,
            label='Ground Truth'
        )

        # --- Prediction 1 ---
        ax.plot(
            time, x1_np[:, d],
            linestyle='--',
            linewidth=1.6,
            label=labels[0]
        )

        # --- Prediction 2 ---
        ax.plot(
            time, x2_np[:, d],
            linestyle=':',
            linewidth=1.6,
            label=labels[1]
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"State Dimension {d}")

        ax.legend(frameon=False, ncol=3)
        ax.grid(True, linestyle='--', alpha=0.4)

        if save_path:
            dim_save_path = save_path.replace(".png", f"_dim{d}.png")
            plt.savefig(dim_save_path, dpi=300, bbox_inches="tight")

        plt.close(fig)


def visulization_experts(experts, save_path):
    E = len(experts)
    eigvals_list = [np.linalg.eigvals(K) for K in experts]
    # 计算子图布局
    n_rows = 1
    n_cols = E  # 每个专家一列
    # 创建图形
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for e_idx in range(E):
        position = e_idx + 1
        ax = fig.add_subplot(n_rows, n_cols, position)
        unit_circle = plt.Circle((0, 0), 1.0, color='gray', linestyle='--', fill=False, linewidth=1.5)
        ax.add_artist(unit_circle)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        eigvals = np.asarray(eigvals_list[e_idx])
        ax.scatter(eigvals.real, eigvals.imag, label=f'Expert {e_idx}', alpha=0.75, s=40)
        ax.set_title(f'Expert Koopman Spectra')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal')
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.legend(loc="upper right")
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path)
    plt.close(fig)

def visulization_phase_experts(x, experts, weights, save_path=None, triples=None):
    """
    绘制专家的相图和谱图（分别保存为两张图）单摆专属绘图函数
    x: Tensor [B, T, D_data] - 数据
    experts: list of Tensor [B, T, D_data] - 每个专家的输出
    weights: Tensor [B, T, E] - 专家权重
    triples: list of (which, (d1, d2, d3)) - 只保留"data"类型
    """
    
    # === 数据准备 ===
    B, T, D_data = x.shape
    if isinstance(x, Tensor):
        x_flat = x.reshape(-1, D_data).cpu().detach().numpy()
    else:
        x_flat = x.reshape(-1, D_data)

    E = len(experts)
    eigvals_list = [np.linalg.eig(K) for K in experts]
    # opts = [K for K in experts]
    if triples is None:
        triples = [("data", (0, 1, 2))]
    else:
        triples = [t for t in triples if t[0] == "data"]
    n_triples = len(triples)

    # === 第一张图：每个专家的相图 ===
    n_rows, n_cols = n_triples, E
    fig1 = plt.figure(figsize=(5, 9))

    for e_idx in range(E):
        for i, (_, (d1, d2, d3)) in enumerate(triples):
            ax = fig1.add_subplot(2,E//2,e_idx+1,
                                  projection='3d' if D_data >= 3 else None)

            if D_data >= 3:
                sc = ax.scatter(
                    x_flat[:, d1], x_flat[:, d2], x_flat[:, d3],
                    c=weights[:, e_idx].cpu().detach().numpy() if isinstance(weights, Tensor) else weights[:, e_idx],
                    cmap='plasma', s=10, alpha=0.7
                )
                ax.set_zlabel(f"dim {d3}")
            else:
                sc = ax.scatter(
                    x_flat[:, d1], x_flat[:, d2],
                    c=weights[:, e_idx].cpu().detach().numpy() if isinstance(weights, Tensor) else weights[:, e_idx],
                    cmap='plasma', s=10, alpha=0.7
                )

            ax.set_title(rf"Weight of basis matrix $G_{{{e_idx+1}}}$ ", fontsize=16)
            # ax.set_xlabel(f"dim {d1}")
            # ax.set_ylabel(f"dim {d2}")
            ax.axis('off')
            cbar = fig1.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.6, aspect=25, pad=0.05)
            cbar.outline.set_linewidth(0.5)
            cbar.ax.tick_params(labelsize=10)
            # cbar.outline.set_visible(False)
    plt.subplots_adjust(top=0.95, bottom=0.0, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_phase.png'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # === 第二张图：每个专家的 Koopman 谱 ===\

    siam_blue = "#003366"  # 深普鲁士蓝
    siam_red = "#990000"   # 深红
    siam_gray = "#444444"  # 辅助线灰色

    fig2 = plt.figure(figsize=(5, 9))
    for e_idx in range(E):
        ax = fig2.add_subplot(2, E//2, e_idx+1)
        eigvals, _ = eigvals_list[e_idx]
        opt = experts[e_idx]
        # eigvals = np.asarray(np.exp(eigenvals))
        eigvals = eigvals[np.imag(eigvals) >= -1e-6]
        # 绘制特征值点
        ax.scatter(eigvals.real, eigvals.imag, s=100, edgecolors='k',
                facecolors=siam_blue, linewidths=0.5, alpha=1.0, zorder=4)
        
        # 绘制单位圆
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color=siam_gray, linestyle='-', linewidth=0.8, zorder=1)
        
        # 绘制切线箭头
        arrow_length = 0.18  # 相对单位圆长度
        for i in range(len(eigvals)):
            lam = eigvals[i]
            x, y = lam.real, lam.imag
            
            # 旋转角
            angle = np.angle(lam) / np.pi
            label = f"${angle:.2f}\pi$"
            ax.text(x + 0.03, y + 0.05, label,
                    fontsize=16, family='serif', color=siam_red)

            # lam = eigvals[i]
            # # x, y = lam.real, lam.imag
            # v = eigvecs[:, i]  # 每个特征值对应的特征向量
            
            # vr = np.real(v)
            # vi = np.imag(v)
            
            # # 旋转方向
            print(opt[1,0])
            rotation_sign = np.sign(opt[1,0])
            
            # 单位圆切线箭头
            dx = rotation_sign * (-y)
            dy = rotation_sign * x
            
            norm = np.sqrt(dx**2 + dy**2)
            dx = dx / norm * arrow_length
            dy = dy / norm * arrow_length
            
            ax.arrow(x, y, dx, dy, width=0.008,          # 箭身更粗
                    head_width=0.05,      # 箭头头部更大
                    head_length=0.08,     # 箭头长度加长
                    fc=siam_red, ec=siam_red,
                    zorder=5,           # 放在点上方
                    alpha=0.9)             
        # # 添加特征值角度标签（π为单位）
        # for val in eigvals:
        #     angle = np.angle(val)
        #     label = f"{angle/np.pi:.2f}π"
        #     ax.text(val.real + 0.03, val.imag + 0.05, label,
        #             fontsize=16, fontweight='bold', color="#B33F34", ha='left', va='bottom')
        
        ax.set_title(f'Spectrum of basis matrix $G_{{{e_idx+1}}}$', fontsize=16)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_xlabel('$\mathbb{Re}(\lambda)$', fontsize=12)
        ax.set_ylabel('$\mathbb{Im}(\lambda)$', fontsize=12)
        ax.set_aspect('equal')
        # 移除四周多余边框，保留轴线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # ax.axis('off')
        
    plt.subplots_adjust(top=0.95, bottom=0.0, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
    if save_path:
        plt.savefig(save_path.replace('.png', '_spectrum.png'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig2)

# def visulization_phase_experts(x, experts, weights, save_path=None, triples=None):
#     """
#     绘制专家的相图
#     x: Tensor [B, T, D_data] - 数据
#     experts: list of Tensor [B, T, D_data] - 每个专家的输出
#     weights: Tensor [B, T, E] - 专家权重
#     triples: list of (which, (d1, d2, d3)) - 只保留"data"类型
#     """
    
#     # 合并batch和时间维度
#     B, T, D_data = x.shape
#     x_flat = x.reshape(-1, D_data).cpu().detach().numpy() if isinstance(x, Tensor) else x.reshape(-1, D_data)
#     # weights = weights
#     eigvals_list = [np.linalg.eigvals(K) for K in experts]
#     E = len(experts)
#     # 筛选只保留"data"的triples
#     if triples is None:
#         triples = [("data", (0, 1, 2))]
#     else:
#         triples = [t for t in triples if t[0] == "data"]
    
#     # 计算子图布局
#     # n_triples = len(triples)
#     # n_rows = 2 * n_triples
#     # n_cols = E  # 每个专家一列
#     n_triples = len(triples)
#     n_rows = 2 * n_triples
#     n_cols = E
#     weight_min = 0
#     weight_max = 1
#     # 绘制切线箭头
#     arrow_length = 0.18  
    
    
#     # 创建图形
#     fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
#     all_axes = []
#     # cax_weight = fig.add_axes([0.1, 0.08, 0.8, 0.02])  # 实部colorbar位置：[left, bottom, width, height]
#     # 为每个专家创建子图
#     for e_idx in range(E):
#         opt = experts[e_idx]
#         for i, (_, dims) in enumerate(triples):
#             d1, d2 = dims[0], dims[1]
#             d3 = dims[2] if len(dims) > 2 else None
#             # 计算正确的子图位置
#             row_weight = 2 * i
#             pos_weight = row_weight * n_cols + e_idx + 1
#             use_3d = (D_data >= 3) and (d3 is not None) and (d3 < D_data)
            
#             if use_3d:
#                 ax_w = fig.add_subplot(n_rows, n_cols, pos_weight, projection="3d")
#                 scatter_ref = ax_w.scatter(
#                     x_flat[:, d1],
#                     x_flat[:, d2],
#                     x_flat[:, d3],
#                     c=weights[:, e_idx],
#                     cmap="plasma",
#                     s=4,
#                     alpha=0.7,
#                     # vmin=weight_min,
#                     # vmax=weight_max,
#                 )
#                 ax_w.set_zlabel(f"dim {d3}", fontsize=10)
#             else:
#                 ax_w = fig.add_subplot(n_rows, n_cols, pos_weight)
#                 scatter_ref = ax_w.scatter(
#                     x_flat[:, d1],
#                     x_flat[:, d2],
#                     c=weights[:, e_idx],
#                     cmap="plasma",
#                     s=6,
#                     alpha=0.7,
#                     # vmin=weight_min,
#                     # vmax=weight_max,
#                 )

#             ax_w.set_title(
#                 rf"Weight of basis matrix $\mathrm{{{e_idx+1}}}$",
#                 fontsize=16
#             )
#             ax_w.tick_params(labelsize=9)
#             all_axes.append(ax_w)
            
#             # # 设置标签和标题
#             # ax.set_title(f"Weight of basis matrix")
#             fig.colorbar(scatter_ref, ax=ax_w,  orientation='horizontal', shrink=0.6, aspect=25, pad=0.05)
#             # ax.axis('off')
#             row_spec = 2 * i + 1
#             pos_spec = row_spec * n_cols + e_idx + 1

#             ax_s = fig.add_subplot(n_rows, n_cols, pos_spec)

#             unit_circle = plt.Circle(
#                 (0, 0),
#                 1.0,
#                 color=colors['siam_gray'],
#                 fill=False,
#                 linestyle='-',
#                 linewidth=0.8,
#                 zorder=1,
#             )
#             ax_s.add_artist(unit_circle)

#             eigvals = np.asarray(eigvals_list[e_idx])
#             for i in range(len(eigvals)):
#                 lam = eigvals[i]
#                 x, y = lam.real, lam.imag
#                 ax_s.scatter(
#                     x,
#                     y,
#                     s=80,
#                     edgecolors='k',
#                     facecolors=colors['siam_blue'],
#                     linewidths=0.5,
#                     alpha=1.0,
#                     zorder=4,
#                 )
#                 # rotation_sign = np.sign(opt[1,0])
                
#                 # # 单位圆切线箭头
#                 # dx = rotation_sign * (-y)
#                 # dy = rotation_sign * x
                
#                 # norm = np.sqrt(dx**2 + dy**2)
#                 # dx = dx / norm * arrow_length
#                 # dy = dy / norm * arrow_length
                
#                 # ax_s.arrow(x, y, dx, dy, width=0.008,          # 箭身更粗
#                 #         head_width=0.05,      # 箭头头部更大
#                 #         head_length=0.08,     # 箭头长度加长
#                 #         fc=colors['siam_red'], ec=colors['siam_red'],
#                 #         zorder=5,           # 放在点上方
#                 #         alpha=0.9)  
                       
#             ax_s.set_title(
#                 rf'Spectrum of basis matrix $\mathrm{{{e_idx+1}}}$',
#                 fontsize=16,
#             )

#             ax_s.set_xlim([-1.65, 1.65])
#             ax_s.set_ylim([-1.2, 1.2])
#             ax_s.set_aspect('equal')

#             ax_s.set_xlabel(r'$\mathbb{Re}(\lambda)$', fontsize=12)
#             ax_s.set_ylabel(r'$\mathbb{Im}(\lambda)$', fontsize=12)
#             ax_s.tick_params(labelsize=9)

#             ax_s.spines['top'].set_visible(False)
#             ax_s.spines['right'].set_visible(False)
#             ax_s.xaxis.set_ticks_position('bottom')
#             ax_s.yaxis.set_ticks_position('left')

#             all_axes.append(ax_s)

#     # 保证上下左右间距一致，使weight图和谱图严格对齐
#     fig.subplots_adjust(
#         top=0.92,
#         bottom=0.12,
#         left=0.06,
#         right=0.98,
#         hspace=0.35,
#         wspace=0.25,
#     )
#     # 调整布局并保存
#     # plt.tight_layout(rect=[0, 0.1, 1, 1])
#     # plt.subplots_adjust(top=0.95, bottom=0.0, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
#     # plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close(fig)

def visulization_phase_spectra(x, koopman_matrix, save_path=None, triples=None):
    """
    绘制相图与谱图结合的图
    x: Tensor [B, T, D_data] - 数据
    spectrum: Tensor [B, T, S] - 复数谱值（S是谱的数量）
    triples: list of (which, (d1, d2, d3)) - 只保留"data"类型
    """
    spectrum = np.linalg.eigvals(koopman_matrix)
    # real_spec, imag_spec = spectrum.real, spectrum.imag
    # 合并batch和时间维度
    B, T, D_data = x.shape
    # _, _, S = spectrum.shape
    # _, D_data = x.shape
    _, S = spectrum.shape
    x_flat = x.reshape(-1, D_data).cpu().detach().numpy() if isinstance(x, Tensor) else x.reshape(-1, D_data)
    spectrum_flat = spectrum.reshape(-1, S)
    # import pdb; pdb.set_trace()
    # print(spectrum_flat[-10:])
    # 筛选只保留"data"的triples
    if triples is None:
        triples = [("data", (0, 1, 2))]
    else:
        triples = [t for t in triples if t[0] == "data"]
    
    # 计算子图布局
    n_triples = len(triples)
    n_rows = 2 * n_triples  # 两行：实部和虚部
    n_cols = max(S, 1)  # 至少1列
    # 计算全局颜色范围（实部和虚部分开）
    real_max = np.max(np.abs(spectrum_flat.real))
    real_min = -np.max(np.abs(spectrum_flat.real))
    imag_max = np.max(np.abs(spectrum_flat.imag))
    imag_min = -np.max(np.abs(spectrum_flat.imag))

    # 创建图形
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    # 创建共享的colorbar轴
    cax_real = fig.add_axes([0.92, 0.55, 0.02, 0.3])  # 实部colorbar位置
    cax_imag = fig.add_axes([0.92, 0.15, 0.02, 0.3])  # 虚部colorbar位置
    # 为每个谱的实部和虚部分别创建子图
    for s_idx in range(S):
        # 获取当前谱的实部和虚部
        spec_real = spectrum_flat[:, s_idx].real
        spec_imag = spectrum_flat[:, s_idx].imag
        
        # 创建实部行
        for i, (_, (d1, d2, d3)) in enumerate(triples):
            # 计算正确的子图位置
            position = 2 * s_idx * n_triples + i + 1
            
            # 创建子图
            if D_data >= 3 and d3 < D_data:
                ax_real = fig.add_subplot(n_rows, n_cols, position, projection='3d')
            else:
                ax_real = fig.add_subplot(n_rows, n_cols, position)
            
            # 绘制3D或2D图
            if D_data >= 3:
                sc_real = ax_real.scatter(
                    x_flat[:, d1], x_flat[:, d2], x_flat[:, d3],
                    c=spec_real, cmap='coolwarm', s=4,
                    vmin=real_min,  # 使用实部全局最小值
                    vmax=real_max   # 使用实部全局最大值
                )
                ax_real.set_zlabel(f"dim {d3}")
            else:
                sc_real = ax_real.scatter(
                    x_flat[:, d1], x_flat[:, d2],
                    c=spec_real, cmap='coolwarm', s=4,
                    vmin=real_min,  # 使用实部全局最小值
                    vmax=real_max   # 使用实部全局最大值
                )
            
            # 设置标签和标题
            ax_real.set_title(f"Spectrum {s_idx} Real - Dims {d1},{d2}" + (f",{d3}" if D_data >= 3 else ""))
            ax_real.set_xlabel(f"dim {d1}")
            ax_real.set_ylabel(f"dim {d2}")

        
        # 创建虚部行
        for i, (_, (d1, d2, d3)) in enumerate(triples):
            position = (2 * s_idx + 1) * n_triples + 1
            
            # 创建子图
            if D_data >= 3 and d3 < D_data:
                ax_imag = fig.add_subplot(n_rows, n_cols, position, projection='3d')
            else:
                ax_imag = fig.add_subplot(n_rows, n_cols, position)
            
            # 绘制3D或2D图
            if D_data >= 3:
                sc_imag = ax_imag.scatter(
                    x_flat[:, d1], x_flat[:, d2], x_flat[:, d3],
                    c=spec_imag, cmap='coolwarm', s=4,
                    vmin=imag_min,  # 使用虚部全局最小值
                    vmax=imag_max   # 使用虚部全局最大值
                )
                ax_imag.set_zlabel(f"dim {d3}")
            else:
                sc_imag = ax_imag.scatter(
                    x_flat[:, d1], x_flat[:, d2],
                    c=spec_imag, cmap='coolwarm', s=4,
                    vmin=imag_min,  # 使用虚部全局最小值
                    vmax=imag_max   # 使用虚部全局最大值
                )
            
            # 设置标签和标题
            ax_imag.set_title(f"Spectrum {s_idx} Imag - Dims {d1},{d2}" + (f",{d3}" if D_data >= 3 else ""))
            ax_imag.set_xlabel(f"dim {d1}")
            ax_imag.set_ylabel(f"dim {d2}")
    # 添加共享的colorbar
    fig.colorbar(sc_real, cax=cax_real, label='Real Value')
    fig.colorbar(sc_imag, cax=cax_imag, label='Imaginary Value')
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    # path = f"./figures/train/{self.args.setting}/phase_plot{batch_idx}.pdf"
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_koopman_moe_diagnostics(koopmans, eigvals_list, gate_probs=None, block_idx=None, save_path=None):
    B, T, E = gate_probs.shape
    # import pdb; pdb.set_trace()  # 调试用
    # print(gate_probs.shape)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    num_experts = len(eigvals_list)

    # === 子图1: 每个专家的 Koopman 谱 ===
    ax = axs[0]
    unit_circle = plt.Circle((0, 0), 1.0, color='gray', linestyle='--', fill=False, linewidth=1.5)
    ax.add_artist(unit_circle)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    for i, eigvals in enumerate(eigvals_list):
        eigvals = np.asarray(eigvals)
        ax.scatter(eigvals.real, eigvals.imag, label=f'Expert {i}', alpha=0.75, s=40)

    ax.set_title(f'Expert Koopman Spectra')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect('equal')
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.legend(loc="upper right")

    # === 子图2: gate 激活柱状图 ===
    ax = axs[1]
    if gate_probs is not None:
        # gate_probs: [B, T, E]，选取最后一个通道
        gate_prob = gate_probs[0]  # shape: [T, E]

        # 对 batch 求平均，得到最后通道的专家激活情况
        # bars = ax.bar(np.arange(len(gate_prob)), gate_prob, color='cornflowerblue',label='Gate Probability')
        im = ax.imshow(gate_prob.T, aspect='auto', origin='lower', cmap='viridis')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Gate Probability')
        # for i, val in enumerate(avg_gate):
        #     ax.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
        ax.set_title("Gate Activation (mean across batch)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Expert")
        # ax._colorbars(label='Gate Probability')
        ax.set_title(f"Sample {i} Gate Probabilities")
    else:
        avg_gate = np.ones(num_experts) / num_experts
        ax.text(0.1, 0.5, "No gate_probs provided", fontsize=12)

    # === 子图3: 加权融合的 Koopman 谱 ===
    ax = axs[2]

    # 加权融合, koopmans:[E,D,D]
    K_soft = np.einsum('te,edf->tdf', gate_prob, koopmans)  # [T, D, D]
    eigvals = np.linalg.eigvals(K_soft) #[T, D]

    unit_circle = plt.Circle((0, 0), 1.0, color='gray', linestyle='--', fill=False, linewidth=1.5)
    ax.add_artist(unit_circle)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    for d in range(E):
        im = ax.scatter(eigvals[:, d].real, eigvals[:, d].imag, c=np.linspace(0, 1, T), cmap='viridis', s=30, label=f'λ_{d}')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Gate Probability')
    # for i, w in enumerate(weights):
    #     ax.text(1.1, 1.0 - i * 0.1, f"Expert {i}: {w:.2f}", fontsize=9)
    ax.set_title("Weighted Koopman Spectrum")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect('equal')
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # print(f"[✔] Koopman diagnostics saved to {save_path}")
    plt.close()


def get_next_indexed_filename(prefix, save_dir, suffix=".pdf"):
    files = glob.glob(os.path.join(save_dir, f"{prefix}_*.pdf"))
    indices = [
        int(f.split("_")[-1].split(".")[0])
        for f in files
        if f.split("_")[-1].split(".")[0].isdigit()
    ]
    next_index = max(indices) + 1 if indices else 0
    return os.path.join(save_dir, f"{prefix}_{next_index}{suffix}"), next_index

def visualize_koopman_spectrum_hook(block_idx=0, save_dir='./koopman_spectrum'):
    os.makedirs(save_dir, exist_ok=True)
    
    counter = [0]  # 使用列表作为可变闭包变量

    def hook_fn(module, input, output):
        counter[0] += 1  # 每调用一次 hook 累加
        
        if counter[0] % 50 != 0:
            return  # 只在每 50 次保存一次图像

        koopmans = [
            expert.get_koopman_matrix().detach().cpu().numpy()
            for expert in module.experts
        ]
        eigvals_list = [np.linalg.eigvals(K) for K in koopmans]
        
        gate_probs = output[1] if isinstance(output, tuple) and len(output) > 1 else None

        prefix = f"koopman_block_{block_idx}"
        save_path, _ = get_next_indexed_filename(prefix, save_dir)
        
        # 绘图和保存
        plot_koopman_moe_diagnostics(
            koopmans, eigvals_list, gate_probs, 
            block_idx=block_idx, save_path=save_path
        )
    
    return hook_fn

def visualize_eigenfunction(zs, save_path=None):
    """
    可视化多个复值 Koopman 特征函数在复平面上的结构
    每个特征函数绘制为 (Re φ, Im φ) 散点图，颜色表示幅值或相位。
    
    参数:
        x: 原始输入 (B, T, D)，仅用于维度匹配
        zs: Koopman 特征函数输出 (B, T, feature_dim)，其中前一半为实部，后一半为虚部
        save_path: 保存路径（可选）
    """

    # 展平输入
    zs_flat = zs.reshape(-1, zs.shape[-1]).cpu().detach().numpy() if hasattr(zs, 'cpu') else zs.reshape(-1, zs.shape[-1])

    # 检查特征维度
    feature_dim = zs_flat.shape[1]
    if feature_dim % 2 != 0:
        logger.warning("特征维度 %d 不是偶数，无法完全配对为复数特征函数，截断最后一维", feature_dim)
        feature_dim -= 1
        zs_flat = zs_flat[:, :feature_dim]
    num_complex_features = feature_dim // 2

    # 计算复数特征函数
    fig, axes = plt.subplots(1, num_complex_features, figsize=(6 * num_complex_features, 10))
    if num_complex_features == 1:
        axes = [axes]

    for i in range(num_complex_features):
        real_part = zs_flat[:, i]
        imag_part = zs_flat[:, i + num_complex_features]
        # magnitude = np.sqrt(real_part**2 + imag_part**2)
        phase = np.arctan2(imag_part, real_part)

        ax = axes[i]
        sc = ax.scatter(real_part, imag_part, c=phase, cmap='twilight', s=10, alpha=0.7)
        ax.set_title(f"$\\varphi_{i+1}$", fontsize=12)
        ax.set_xlabel("Re($\\varphi$)")
        ax.set_ylabel("Im($\\varphi$)")
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", 
                               shrink=0.5, aspect=20, pad=0.01)
        cbar.outline.set_visible(False)
        # cbar.set_label("相位 (radians)")

        # 可选: 画出单位圆作为参考
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.6)

    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("图像已保存到: %s", save_path)
    else:
        plt.show()



# def visualize_mag_phase(x, zs, save_path=None):
#     """
#     SIAM 风格局部优化版：不修改全局 rcParams
#     """
#     B, T, D = x.shape
#     x_flat = x.reshape(-1, D).cpu().detach().numpy() if hasattr(x, 'cpu') else x.reshape(-1, D)
#     zs_flat = zs.reshape(-1, zs.shape[-1]).cpu().detach().numpy() if hasattr(zs, 'cpu') else zs.reshape(-1, zs.shape[-1])

#     feature_dim = zs_flat.shape[1]
#     if feature_dim % 2 != 0:
#         feature_dim -= 1
#         zs_flat = zs_flat[:, :feature_dim]

#     num_features = feature_dim // 2
#     magnitudes, phases = [], []
#     for i in range(num_features):
#         r, im = zs_flat[:, i], zs_flat[:, i + num_features]
#         magnitudes.append(np.sqrt(r**2 + im**2))
#         phases.append(np.arctan2(im, r))

#     mag_min, mag_max = np.min(magnitudes), np.max(magnitudes)
    
#     # 使用局部 style context 仅在此代码块内生效
#     # 'serif' 确保数学公式和文字使用衬线体
#     with plt.style.context(['seaborn-v0_8-paper', {'font.family': 'serif'}]):
#         fig = plt.figure(figsize=(4 * num_features + 1, 8))
#         gs = fig.add_gridspec(2, num_features + 1, width_ratios=[1]*num_features + [0.08], 
#                               wspace=0.15, hspace=0.3)

#         for i in range(num_features):
#             # --- Row 0: Magnitude (Viridis) ---
#             ax_m = fig.add_subplot(gs[0, i], projection='3d' if D == 3 else None)
#             im_mag = ax_m.scatter(x_flat[:, 0], x_flat[:, 1], 
#                                   *( [x_flat[:, 2]] if D == 3 else [] ),
#                                   c=magnitudes[i], cmap='viridis', 
#                                   vmin=mag_min, vmax=mag_max, s=5, alpha=0.7)
            
#             # 局部设置标题，使用 LaTeX
#             ax_m.set_title(rf'Magnitude $|\psi_{{{i+1}}}(\mathbf{{x}})|$', fontsize=13)
#             ax_m.axis('off')

#             # --- Row 1: Phase (Twilight) ---
#             ax_p = fig.add_subplot(gs[1, i], projection='3d' if D == 3 else None)
#             im_phase = ax_p.scatter(x_flat[:, 0], x_flat[:, 1], 
#                                     *( [x_flat[:, 2]] if D == 3 else [] ),
#                                     c=phases[i], cmap='twilight', 
#                                     vmin=-np.pi, vmax=np.pi, s=5, alpha=0.7)
            
#             ax_p.set_title(rf'Phase $\mathrm{{arg}}(\psi_{{{i+1}}}(\mathbf{{x}}))$', fontsize=13)
#             ax_p.axis('off')

#         # --- Colorbars ---
#         # 显式设置 Colorbar 边框线宽，体现 SIAM 的细致感
#         ax_cbar_mag = fig.add_subplot(gs[0, -1])
#         ax_cbar_mag.axis('off')
#         cax_m = inset_axes(ax_cbar_mag, width="25%", height="70%", loc='center')
#         cb_m = fig.colorbar(im_mag, cax=cax_m)
#         cb_m.outline.set_linewidth(0.5)
#         cb_m.ax.tick_params(labelsize=9)

#         ax_cbar_p = fig.add_subplot(gs[1, -1])
#         ax_cbar_p.axis('off')
#         cax_p = inset_axes(ax_cbar_p, width="25%", height="70%", loc='center')
#         cb_p = fig.colorbar(im_phase, cax=cax_p)
#         cb_p.set_ticks([-np.pi, 0, np.pi])
#         cb_p.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
#         cb_p.outline.set_linewidth(0.5)
#         cb_p.ax.tick_params(labelsize=9)

#         if save_path:
#             # 显式设置白色背景，不依赖全局 transparent 状态
#             fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
        
#         plt.show()

def visualize_mag_phase(x, zs, save_path=None):
    # === 数据处理部分 (保持逻辑) ===
    B, T, D = x.shape
    x_flat = x.reshape(-1, D).cpu().detach().numpy() if hasattr(x, 'cpu') else x.reshape(-1, D)
    zs_flat = zs.reshape(-1, zs.shape[-1]).cpu().detach().numpy() if hasattr(zs, 'cpu') else zs.reshape(-1, zs.shape[-1])
    
    feature_dim = zs_flat.shape[1]
    if feature_dim % 2 != 0:
        feature_dim = feature_dim - 1
        zs_flat = zs_flat[:, :feature_dim]
    
    num_complex_features = feature_dim // 2
    magnitudes, phases = [], []
    for i in range(num_complex_features):
            real, imag = zs_flat[:, i], zs_flat[:, i + num_complex_features]
            magnitudes.append(np.sqrt(real**2 + imag**2))
            phases.append(np.arctan2(imag, real))

    vmin, vmax = np.min(magnitudes), np.max(magnitudes)

    # === 绘图部分 (SIAM 风格 + 底部 Colorbar) ===
    # 使用局部 style context 确保衬线字体和学术风格，不影响全局
    with plt.style.context(['seaborn-v0_8-paper', {'font.family': 'serif'}]):
        fig = plt.figure(figsize=(5 * num_complex_features, 9))
        
        for i in range(num_complex_features):
            # --- 第一行：幅值 (Magnitude) ---
            ax_mag = fig.add_subplot(2, num_complex_features, i+1, projection='3d' if D == 3 else None)
            im_mag = ax_mag.scatter(x_flat[:,0], x_flat[:,1], *([x_flat[:,2]] if D == 3 else []),
                                   s=10, c=magnitudes[i], cmap='viridis', alpha=0.7,
                                   vmax=vmax, vmin=vmin, edgecolors='none')
            
            ax_mag.set_title(rf'Amplitude $\Vert\mathbf{{z}}_{{{i+1}}}\Vert$', fontsize=16, pad=10)
            ax_mag.axis('off')
            
            # 底部 Colorbar: horizontal + pad
            cb1 = fig.colorbar(im_mag, ax=ax_mag, **CBAR_OPTS)
            cb1.outline.set_linewidth(0.5)

            # --- 第二行：相位 (Phase) ---
            ax_phase = fig.add_subplot(2, num_complex_features, i+1+num_complex_features, 
                                      projection='3d' if D == 3 else None)
            im_phase = ax_phase.scatter(x_flat[:,0], x_flat[:,1], *([x_flat[:,2]] if D == 3 else []),
                                       s=10, c=phases[i], cmap='twilight', alpha=0.7, 
                                       vmin=-np.pi, vmax=np.pi, edgecolors='none')
            
            ax_phase.set_title(rf'Phase $\mathrm{{arg}}(\mathbf{{z}}_{{{i+1}}})$', fontsize=16, pad=10)
            ax_phase.axis('off')
            
            # 底部相位 Colorbar
            cb2 = fig.colorbar(im_phase, ax=ax_phase, **CBAR_OPTS)
            cb2.set_ticks([-np.pi, 0, np.pi])
            cb2.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
            cb2.outline.set_linewidth(0.5)

        # 核心：使用明确的间距控制，确保横向拼接是对齐的
        plt.subplots_adjust(top=0.95, bottom=0.0, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
        if save_path:
            # SIAM 风格建议使用白色背景而非透明，确保打印清晰度
            fig.savefig(save_path, bbox_inches='tight', facecolor='white', transparent=False, dpi=300)
        
        plt.show()

    return {
        'magnitudes': magnitudes,
        'phases': phases,
        'num_complex_features': num_complex_features,
        'x_flat': x_flat
    }
# def visualize_mag_phase(x, zs, save_path=None):
#     B, T, D = x.shape
#     assert D <= 3
    
#     # 将输入数据展平
#     x_flat = x.reshape(-1, D).cpu().detach().numpy() if hasattr(x, 'cpu') else x.reshape(-1, D)
#     zs_flat = zs.reshape(-1, zs.shape[-1]).cpu().detach().numpy() if hasattr(zs, 'cpu') else zs.reshape(-1, zs.shape[-1])
    
#     # 获取特征维度
#     feature_dim = zs_flat.shape[1]
    
#     # 检查特征维度是否为偶数（因为需要配对为复数）
#     if feature_dim % 2 != 0:
#         print(f"警告: 特征维度 {feature_dim} 不是偶数，无法完全配对为复数特征函数")
#         # 这里我们可以选择只使用前 feature_dim-1 个维度，或者采用其他处理方式
#         # 为了简单起见，我们使用前 feature_dim-1 个维度
#         feature_dim = feature_dim - 1
#         zs_flat = zs_flat[:, :feature_dim]
    
#     # 计算复数特征函数的数量
#     num_complex_features = feature_dim // 2
    
#     # 将实数特征函数转换为复数特征函数
#     magnitudes = []
#     phases = []
    
#     for i in range(num_complex_features):
#         real_part = zs_flat[:, i]                    # 实部
#         imag_part = zs_flat[:, i + num_complex_features]  # 虚部
#         # real_part = zs_flat[:, 2*i]                    # 实部
#         # imag_part = zs_flat[:, 2*i + 1]  # 虚部
        
#         # 计算复数特征函数的幅值和相位
#         magnitude = np.sqrt(real_part**2 + imag_part**2)
#         phase = np.arctan2(imag_part, real_part)  # 注意：arctan2(imag, real)
        
#         magnitudes.append(magnitude)
#         phases.append(phase)

#     vmin, vmax = np.min(magnitudes), np.max(magnitudes)
#     # 创建图形 - 2行，num_complex_features列
#     fig = plt.figure(figsize=(6 * num_complex_features, 10))
    
#     # 为每个复数特征函数绘制幅值和相位
#     for i in range(num_complex_features):
#         # 第一行：幅值
#         ax_mag = fig.add_subplot(2, num_complex_features, i+1, projection='3d' if D == 3 else None)
#         if D == 3:
#             im_mag = ax_mag.scatter(x_flat[:,0], x_flat[:,1], x_flat[:,2], 
#                                    s=8, c=magnitudes[i], cmap='turbo', alpha=0.7,
#                                    vmax=vmax, vmin=vmin)
#         else:
#             im_mag = ax_mag.scatter(x_flat[:,0], x_flat[:,1], 
#                                    s=8, c=magnitudes[i], cmap='turbo', alpha=0.7,
#                                    vmax=vmax, vmin=vmin)
        
#         # ax_mag.set_title(f'$\\vert \\varphi_{i+1}\\vert$', fontsize=20, fontweight='bold')
#         ax_mag.set_title(f'Amplitude $\\vert\\varphi\\vert$', fontsize=18, fontweight='bold')
#         ax_mag.axis('off')
        
#         # 添加colorbar
#         cbar_mag = fig.colorbar(im_mag, ax=ax_mag, orientation="horizontal", 
#                                shrink=0.5, aspect=20, pad=0.01)
#         cbar_mag.outline.set_visible(False)
        
#         # 第二行：相位
#         ax_phase = fig.add_subplot(2, num_complex_features, i+1+num_complex_features, projection='3d' if D == 3 else None)
#         if D == 3:
#             im_phase = ax_phase.scatter(x_flat[:,0], x_flat[:,1], x_flat[:,2], 
#                                        s=8, c=phases[i],  cmap='turbo', alpha=0.7, 
#                                        vmin=-np.pi, vmax=np.pi)
#         else:
#             im_phase = ax_phase.scatter(x_flat[:,0], x_flat[:,1], 
#                                        s=8, c=phases[i],  cmap='turbo', alpha=0.7,
#                                        vmin=-np.pi, vmax=np.pi)
        
#         # ax_phase.set_title(f'$\\angle\\varphi_{i+1}$', fontsize=20, fontweight='bold')
#         ax_phase.set_title(f'Phase $\\angle\\varphi$', fontsize=18, fontweight='bold')
#         ax_phase.axis('off')
        
#         # 添加相位colorbar
#         cbar_phase = fig.colorbar(im_phase, ax=ax_phase, orientation="horizontal", 
#                                  shrink=0.5, aspect=20, pad=0.01)
#         cbar_phase.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
#         cbar_phase.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
#         cbar_phase.outline.set_visible(False)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', transparent=True, dpi=300)
    
#     plt.show()
    
#     # 返回处理后的数据供进一步分析
#     return {
#         'magnitudes': magnitudes,
#         'phases': phases,
#         'num_complex_features': num_complex_features,
#         'x_flat': x_flat
#     }

# def visualize_mag_phase(x, zs, save_path=None):
#     B, T, D = x.shape
#     assert D<=3
#     x_flat = x.reshape(-1, D).cpu().detach().numpy() if isinstance(x, Tensor) else x.reshape(-1, D)
#     magnitude = np.norm(zs)
#     phase = np.arctan2(zs[...,0],zs[...,1])
#     magnitude_flat = magnitude.reshape(-1).cpu().detach().numpy()
#     phase_flat = phase.reshape(-1).cpu().detach().numpy()
#     fig = plt.figure(figsize=(6, 10))
#     # fig, axes = plt.subplots(2, 1, figsize=(6, 10), projection='3d' if D == 3 else None)

#     # 幅值
#     ax_mag = fig.add_subplot(2,1,1,projection='3d' if D == 3 else None)
#     if D == 3:
#         im0 = ax_mag.scatter(x_flat[:,0], x_flat[:,1], x_flat[:,2], s=8, c=magnitude_flat, cmap='turbo',alpha=0.5)
#     else:
#         im0 = ax_mag.scatter(x_flat[:,0], x_flat[:,1], s=8, c=magnitude_flat, cmap='turbo')
#     ax_mag.set_title(r"$\Vert\varphi\Vert_2^2$", fontsize=20, fontweight='bold')
#     # ax_mag.set_xlabel(r"$\varphi_1^2+\varphi_2^2$")
#     ax_mag.axis('off')
#     cbar1 = fig.colorbar(im0, ax=ax_mag, orientation='horizontal', 
#                          shrink=0.7,  aspect=40, pad=0.01)
#     cbar1.outline.set_visible(False)
#     # 相位
#     ax_phase = fig.add_subplot(2,1,2,projection='3d' if D == 3 else None)
#     if D == 3:
#         im1 = ax_mag.scatter(x_flat[:,0], x_flat[:,1], x_flat[:,2], s=8, c=magnitude_flat, cmap='turbo',alpha=0.5)
#     else:
#         im1 = ax_mag.scatter(x_flat[:,0], x_flat[:,1], s=8, c=magnitude_flat, cmap='turbo')
    
#     ax_phase.set_title(r"$\tan^{-1} (\varphi_1/\varphi_2)$", fontsize=20, fontweight='bold')
#     ax_phase.set_xlabel(r"$\tan^{-1} (\varphi_1/\varphi_2)$")
#     ax_phase.axis('off')
#     cbar2 = fig.colorbar(im1, ax=ax_phase,orientation='horizontal', 
#                         shrink=0.5,  aspect=40, pad=0.01)
#     # 设置 colorbar 的刻度位置和值
#     cbar2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
#     cbar2.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
#     cbar2.outline.set_visible(False)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight',transparent=True)
#     plt.show()

def visualize_eigenfun(x, eigen, save_path=None):
    """
    SIAM 风格特征函数 (Real/Imag) 绘图函数
    """
    B, T, D = x.shape
    assert D <= 3
    _, _, E = eigen.shape
    
    # 数据转换
    x_flat = x.reshape(-1, D).cpu().detach().numpy() if isinstance(x, Tensor) else x.reshape(-1, D)
    
    # 局部风格设置，避免干扰全局环境
    with plt.style.context(['seaborn-v0_8-paper', {'font.family': 'serif'}]):
        # 建立画布，SIAM 风格不建议过宽
        fig = plt.figure(figsize=(5 * (E // 2), 9))
        
        for j in trange(E):
            # 创建子图
            ax = fig.add_subplot(2, E // 2, j + 1, projection='3d' if D == 3 else None)
            eigen_flat = eigen[..., j].reshape(-1).cpu().detach().numpy()
    

            if D == 3:
                im = ax.scatter(x_flat[:, 0], x_flat[:, 1], *([x_flat[:, 2]] if D == 3 else []),
                            s=10, c=eigen_flat, cmap='RdBu_r', alpha=0.7, edgecolors='none')
            else:
                im = ax.scatter(x_flat[:, 0], x_flat[:, 1], *([x_flat[:, 2]] if D == 3 else []),
                            s=10, c=eigen_flat, cmap='RdBu_r', alpha=0.7, edgecolors='none')
            
            # 设置符合 SIAM 规范的 LaTeX 标题
            title_str = r"$\mathrm{Re}(\mathbf{z})$" if j % 2 == 0 else r"$\mathrm{Im}(\mathbf{z})$"
            ax.set_title(title_str, fontsize=16, pad=10)
            ax.axis('off')
            
            cb = fig.colorbar(im, ax=ax, **CBAR_OPTS)
            cb.outline.set_linewidth(0.5)
        # 核心：使用完全相同的间距参数
        plt.subplots_adjust(top=0.95, bottom=0.0, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
        # plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    # plt.show()

def visulize_latent(x, z, save_path=None):
    B, T, D = x.shape
    assert D<=3
    idx = np.arange(0, B*T, 1000)
    x_flat = x.reshape(-1, D).cpu().detach().numpy() if isinstance(x, Tensor) else x.reshape(-1, D)
    z_flat = z.reshape(-1, D).cpu().detach().numpy() if isinstance(x, Tensor) else z.reshape(-1, D)
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig = plt.figure(figsize=(6, 10))
    axes = []

    ax_phase = fig.add_subplot(2, 1, 1, projection='3d' if D == 3 else None)
    axes.append(ax_phase)
    # 幅值
    if D == 3:
        im0 = ax_phase.scatter(x_flat[idx,0], x_flat[idx,1], x_flat[idx,2], s=6, cmap='RdBu_r')
    else:
        im0 = ax_phase.scatter(x_flat[:,0], x_flat[:,1], s=6, cmap='RdBu_r')
    # ax_phase.set_title(rf"$\varphi_{j+1}$", fontsize=20)
    ax_phase.axis('off')

    ax_latent = fig.add_subplot(2, 1, 1, projection='3d' if D == 3 else None)
    axes.append(ax_latent)
    # 幅值
    if D == 3:
        im0 = ax_latent.scatter(z_flat[idx,0], z_flat[idx,1], z_flat[idx,2], s=6, cmap='RdBu_r')
    else:
        im0 = ax_latent.scatter(z_flat[:,0], z_flat[:,1], s=6, cmap='RdBu_r')
    # ax_latent.set_title(rf"$\varphi_{j+1}$", fontsize=20)
    ax_latent.axis('off')
    # plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.tight_layout()
    # # 在底部单独开一个 [left, bottom, width, height] 位置放 colorbar
    # cbar_ax = fig.add_axes([0.15, 0.04, 0.8, 0.02])  # [x0, y0, w, h] (0~1)
    # cbar = fig.colorbar(im0, cax=cbar_ax, orientation="horizontal",pad=0.15)
    # cbar.set_label("Eigenfunction Value")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.show()