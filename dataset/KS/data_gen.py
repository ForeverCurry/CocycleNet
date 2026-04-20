import numpy as np
from numpy.fft import fft, ifft
from joblib import Parallel, delayed
np.random.seed(2025)

def generate_initial_condition(N:int, initial_scale:float, modes):
    v0 = np.zeros(N, dtype=np.complex128)
    for m in modes:
        if m >= N//2:
            continue
        val = initial_scale * (np.random.randn() + 1j * np.random.randn())
        v0[m] = val
        v0[-m] = np.conj(val)   
    return v0

def ks_etdrk4(L=8*np.pi, N=256, dt=0.1, spin_up_steps =1000, nsteps=500, seed=0, dealias=True):
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)

    # wavenumbers for FFT: k = 2*pi * n / L, where n = 0, 1, ..., N/2-1, -N/2, ..., -1
    k = 2*np.pi * np.fft.fftfreq(N, d=dx)    # real angular wavenumbers

    # linear operator in Fourier space: L(k) = k^2 - k^4
    Lk = k**2 - k**4   # 注意：这是 k^2 - k^4

    # ETDRK4 coefficients
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
    check if the KS trajectory is in a chaotic regime based on spatial standard deviation and spectral entropy.
    
    parameters:
    u: spatial field at a given time step (shape: (N,))
    threshold_std: minimum standard deviation for chaotic regime (KS chaotic states typically have std > 0.5)
    threshold_entropy: minimum spectral entropy for chaotic regime (KS chaotic states typically
    """
    std_val = np.std(u)
    
    # compute Spectral Entropy
    u_hat = np.fft.fft(u)
    psd = np.abs(u_hat[:len(u)//2])**2  # 取正频段功率谱
    psd_norm = psd / np.sum(psd)        # 归一化
    
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
    # KS chaotic states typically have spectral entropy > 0.5 (depends on N and L, but this is a rough threshold)
    is_fine = (std_val > threshold_std) and (spectral_entropy > threshold_entropy)
    
    return is_fine, std_val, spectral_entropy

def safe_ks_etdrk4( **kwargs):
    max_retries = 50
    for attempt in range(max_retries):
        data = ks_etdrk4(**kwargs)
        if not np.isnan(data).any():
            # check if trajectory is chaotic
            passed, s_val, e_val = is_chaotic(data[-1, :])
            
            if passed:
                return data
            else:
                print(f"Warning: Non-chaotic trajectory detected and discarded. Attempts: {attempt}", end='\r')
                
        else:
            print(f"Warning: generate NaN, retry {attempt+1}/{max_retries}...")
    raise RuntimeError(f"retry multiple times and still generate NaN！")

def generate_dataset(num_init=10000, T=100, N=128, L=8*np.pi, dt=0.1, spin_up_steps=20, n_jobs=-1):
    """shape of data: (num_init, T, N)"""
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(safe_ks_etdrk4)(L=L, N=N, dt=dt, spin_up_steps=spin_up_steps, nsteps=T)
        for _ in range(num_init)
    )
    return np.array(results)

def compute_energy(u, L):
    """
    compute energy E(t) = (1/L) * ∫ u(x,t)^2 dx
    """
    N = len(u)
    dx = L / N
    return np.sum(u**2) * dx / L

if __name__ == "__main__":
    L = 8.0 * np.pi       
    N = 128                       
    dt = 0.25
    spin_up_steps = 1000   # warm-up steps to reach chaotic regime
    trajectory_length = 400  # number of time steps to record after spin-up
    batch_size = 20000  
    T = trajectory_length * dt
    ks_data = generate_dataset(num_init = 20000, T = trajectory_length, N = N, L = L, dt = dt, n_jobs = -1) 
    np.save(f'./dataset/KS/KS_8pi.npy', ks_data)
