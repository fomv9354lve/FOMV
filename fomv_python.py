"""
FOMV: Field Operator for Measured Viability - Versión Optimizada y Corregida
Autor: Osvaldo Morales (con asistencia de optimización)
Licencia: AGPL-3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Parámetros del modelo (Tabla 1 del paper)
# =============================================================================
params = {
    'alpha1': 0.1,
    'alpha2': 0.2,
    'delta': 0.05,
    'beta1': 0.3,
    'gamma1': 0.2,
    'gamma2': 0.1,
    'gamma3': 0.1,
    'phi1': 0.3,
    'phi2': 0.2,
    'psi1': 0.2,
    'psi2': 0.2,
    'kappa1': 0.2,
    'kappa2': 0.1,
    'Ec': 0.1,
    'Er': 0.5,
    'Lc': 1.5,
    'Lr': 0.8,
}

# =============================================================================
# Parámetros de simulación REDUCIDOS para prueba rápida
# =============================================================================
sim_params = {
    'sigma': 0.05,
    'Tmax': 50,
    'R': 500,
    'Bgrid': 5,
    'Mgrid': 5,
    'B_range': [0, 1.2],
    'M_range': [0, 0.8],
    'bootstrap_reps': 100,
    'alpha': 0.05,
    'fast_samples': 10,
    'n_cores': mp.cpu_count()
}

# =============================================================================
# Funciones de dinámica (optimizadas para operar con arrays)
# =============================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hard_nonlinear_dynamics_vectorized(x, theta, eta):
    """
    Versión vectorizada: x tiene forma (n_trayectorias, 6)
    eta tiene la misma forma (n_trayectorias, 6)
    Retorna x_star con la misma forma.
    """
    B = x[:, 0]
    M = x[:, 1]
    E = x[:, 2]
    G = x[:, 3]
    T = x[:, 4]
    C = x[:, 5]

    B_star = B + theta['alpha1']*(1 - E) - theta['alpha2']*G
    M_star = (1 - theta['delta'])*M + theta['beta1'] * sigmoid(B - T)
    E_star = E + theta['gamma1']*G - theta['gamma2']*B - theta['gamma3']*M
    G_star = G + theta['phi1']*E - theta['phi2']*(B + M)*(1 - T)
    T_star = T - theta['psi1']*M*(1 - G) + theta['psi2']*G
    C_star = C + theta['kappa1']*T - theta['kappa2']*B

    x_star = np.column_stack([B_star, M_star, E_star, G_star, T_star, C_star]) + eta
    return np.clip(x_star, 0, 1)

def generate_noise_vectorized(sigma, n):
    """
    Genera n vectores de ruido con distribución de aceptación-rechazo.
    Retorna array (n, 6)
    """
    n_needed = n
    collected = []
    total = 0
    while total < n_needed:
        batch_size = min(10000, (n_needed - total) * 2)
        u = np.random.uniform(-1, 1, size=(batch_size, 6))
        prob = np.prod(0.75 * (1 - u**2), axis=1)
        accept = np.random.rand(batch_size) < prob
        accepted = u[accept]
        if len(accepted) > 0:
            collected.append(accepted)
            total += len(accepted)
    noise = np.vstack(collected)[:n_needed]
    return sigma * noise

# =============================================================================
# Funciones de absorción (vectorizadas)
# =============================================================================
def is_collapsed_vectorized(x, theta):
    B = x[:, 0]
    M = x[:, 1]
    E = x[:, 2]
    return (E <= theta['Ec']) | (B + M >= theta['Lc'])

def is_recovered_vectorized(x, theta):
    B = x[:, 0]
    M = x[:, 1]
    E = x[:, 2]
    return (E >= theta['Er']) & (B + M <= theta['Lr'])

# =============================================================================
# Simulación de múltiples trayectorias en paralelo (vectorizada)
# =============================================================================
def simulate_trajectories_vectorized(x0, theta, sigma, Tmax):
    """
    x0: array (n_trayectorias, 6)
    Retorna:
        absorptions: array de strings 'C', 'R' o None
        times: array de tiempos de absorción (Tmax si no hubo)
    """
    n = x0.shape[0]
    x = x0.copy()
    absorptions = np.full(n, None, dtype=object)
    times = np.full(n, Tmax, dtype=int)

    for t in range(Tmax):
        active = (absorptions == None)
        if not np.any(active):
            break

        x_active = x[active]
        collapsed = is_collapsed_vectorized(x_active, theta)
        recovered = is_recovered_vectorized(x_active, theta)

        idx_collapsed = np.where(active)[0][collapsed]
        absorptions[idx_collapsed] = 'C'
        times[idx_collapsed] = t

        idx_recovered = np.where(active)[0][recovered]
        absorptions[idx_recovered] = 'R'
        times[idx_recovered] = t

        still_active = active & (absorptions == None)
        if not np.any(still_active):
            break

        eta = generate_noise_vectorized(sigma, np.sum(still_active))
        x[still_active] = hard_nonlinear_dynamics_vectorized(x[still_active], theta, eta)

    return absorptions, times

# =============================================================================
# Distribución estacionaria de variables rápidas (para proyección)
# =============================================================================
def generate_fast_samples(B, M, theta, sigma, n_samples, burnin=500):
    """
    Genera muestras de las rápidas condicionadas a (B,M) fijos.
    Retorna array (n_samples, 4) con (E,G,T,C)
    """
    samples = []
    fast = np.random.uniform(0, 1, size=4)
    total_steps = n_samples + burnin
    for i in range(total_steps):
        x = np.array([B, M, fast[0], fast[1], fast[2], fast[3]])
        eta = generate_noise_vectorized(sigma, 1)[0]
        x_new = hard_nonlinear_dynamics_vectorized(x.reshape(1, -1), theta, eta.reshape(1, -1))
        fast = x_new[0, 2:]
        if i >= burnin:
            samples.append(fast.copy())
    return np.array(samples)

# =============================================================================
# Cálculo para un punto del grid (B, M) - usado en paralelo
# =============================================================================
def compute_point(BM, theta, sigma, Tmax, R, fast_samples):
    B, M = BM
    try:
        fast_arr = generate_fast_samples(B, M, theta, sigma, fast_samples)
        all_times_C = []
        q_sum = 0.0
        total_traj = 0

        for fast in fast_arr:
            x0 = np.tile(np.array([B, M, fast[0], fast[1], fast[2], fast[3]]), (R, 1))
            absorptions, times = simulate_trajectories_vectorized(x0, theta, sigma, Tmax)

            q_sum += np.sum(absorptions == 'R')
            total_traj += R
            all_times_C.extend(times[absorptions == 'C'])

        q_hat = q_sum / total_traj if total_traj > 0 else np.nan
        mfpt_hat = np.mean(all_times_C) if all_times_C else np.nan
        return (B, M, q_hat, mfpt_hat, all_times_C)

    except Exception as e:
        print(f"Error en punto (B={B:.3f}, M={M:.3f}): {e}")
        raise

# =============================================================================
# Estimación en grid con paralelización
# =============================================================================
def estimate_on_grid_parallel(B_grid, M_grid, theta, sigma, Tmax, R,
                              fast_samples, n_cores):
    points = [(B, M) for B in B_grid for M in M_grid]
    func = partial(compute_point, theta=theta, sigma=sigma,
                   Tmax=Tmax, R=R, fast_samples=fast_samples)
    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap(func, points), total=len(points),
                            desc="Grid points"))

    nB, nM = len(B_grid), len(M_grid)
    Q = np.full((nB, nM), np.nan)
    MFPT = np.full((nB, nM), np.nan)
    times_data = {}
    idx = 0
    for i, B in enumerate(B_grid):
        for j, M in enumerate(M_grid):
            Bres, Mres, q, mfpt, times = results[idx]
            Q[i, j] = q
            MFPT[i, j] = mfpt
            times_data[(i, j)] = times
            idx += 1
    return Q, MFPT, times_data

# =============================================================================
# Bootstrap basado en remuestreo de los tiempos guardados
# =============================================================================
def bootstrap_bands_from_times(times_data, B_grid, M_grid,
                               bootstrap_reps, alpha=0.05):
    nB, nM = len(B_grid), len(M_grid)
    MFPT_hat = np.full((nB, nM), np.nan)
    MFPT_lower = np.full((nB, nM), np.nan)
    MFPT_upper = np.full((nB, nM), np.nan)

    for i in range(nB):
        for j in range(nM):
            times = times_data.get((i, j), [])
            if len(times) == 0:
                continue
            mfpt_hat = np.mean(times)
            MFPT_hat[i, j] = mfpt_hat
            boot_means = []
            for _ in range(bootstrap_reps):
                sample = np.random.choice(times, size=len(times), replace=True)
                boot_means.append(np.mean(sample))
            MFPT_lower[i, j] = np.percentile(boot_means, 100 * alpha / 2)
            MFPT_upper[i, j] = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return MFPT_hat, MFPT_lower, MFPT_upper

# =============================================================================
# Visualización
# =============================================================================
def plot_mfpt(B_grid, M_grid, MFPT, bands_lower=None, title="MFPT on slow manifold"):
    B_mesh, M_mesh = np.meshgrid(B_grid, M_grid, indexing='ij')
    plt.figure(figsize=(8,6))
    contour = plt.contourf(B_mesh, M_mesh, MFPT, levels=20, cmap='viridis')
    plt.colorbar(contour, label='MFPT')
    plt.xlabel('Backlog B')
    plt.ylabel('Memory M')
    plt.title(title)
    if bands_lower is not None:
        plt.contour(B_mesh, M_mesh, MFPT - bands_lower, levels=[0],
                    colors='red', linewidths=2, linestyles='--')
    plt.show()

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("FOMV: Simulación del modelo HARD-nonlinear (Versión Optimizada y Corregida)")
    print("="*60)

    B_grid = np.linspace(sim_params['B_range'][0], sim_params['B_range'][1],
                         sim_params['Bgrid'])
    M_grid = np.linspace(sim_params['M_range'][0], sim_params['M_range'][1],
                         sim_params['Mgrid'])

    print(f"\nUsando {sim_params['n_cores']} núcleos en paralelo.")
    print("Estimando MFPT en grid (esto tomará algunos minutos)...")

    Q, MFPT, times_data = estimate_on_grid_parallel(
        B_grid, M_grid, params, sim_params['sigma'],
        sim_params['Tmax'], sim_params['R'],
        sim_params['fast_samples'], sim_params['n_cores']
    )

    print("\nCalculando bandas bootstrap mediante remuestreo...")
    MFPT_hat, MFPT_lower, MFPT_upper = bootstrap_bands_from_times(
        times_data, B_grid, M_grid,
        sim_params['bootstrap_reps'], sim_params['alpha']
    )

    plot_mfpt(B_grid, M_grid, MFPT_hat, bands_lower=MFPT_lower,
              title="MFPT con banda de protección (95%)")

    print("\nSimulación completada.")
