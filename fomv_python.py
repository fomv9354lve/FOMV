"""
FOMV: Field Operator for Measured Viability
Código de simulación para el modelo HARD-nonlinear (Python)
Autor: Osvaldo Morales
Licencia: AGPL-3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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

sim_params = {
    'sigma': 0.05,
    'Tmax': 200,
    'R': 10000,
    'Bgrid': 20,
    'Mgird': 20,
    'B_range': [0, 1.2],
    'M_range': [0, 0.8],
    'bootstrap_reps': 1000,
    'alpha': 0.05,
}

# =============================================================================
# Funciones de dinámica
# =============================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hard_nonlinear_dynamics(x, theta, eta):
    B, M, E, G, T, C = x
    B_star = B + theta['alpha1']*(1 - E) - theta['alpha2']*G
    M_star = (1 - theta['delta'])*M + theta['beta1'] * sigmoid(B - T)
    E_star = E + theta['gamma1']*G - theta['gamma2']*B - theta['gamma3']*M
    G_star = G + theta['phi1']*E - theta['phi2']*(B + M)*(1 - T)
    T_star = T - theta['psi1']*M*(1 - G) + theta['psi2']*G
    C_star = C + theta['kappa1']*T - theta['kappa2']*B
    x_star = np.array([B_star, M_star, E_star, G_star, T_star, C_star]) + eta
    return np.clip(x_star, 0, 1)

def generate_noise(sigma):
    while True:
        u = np.random.uniform(-1, 1, size=6)
        prob = np.prod(0.75*(1 - u**2))
        if np.random.rand() < prob:
            return sigma * u

# =============================================================================
# Funciones de absorción
# =============================================================================
def is_collapsed(x, theta):
    B, M, E, _, _, _ = x
    return (E <= theta['Ec']) or (B + M >= theta['Lc'])

def is_recovered(x, theta):
    B, M, E, _, _, _ = x
    return (E >= theta['Er']) and (B + M <= theta['Lr'])

# =============================================================================
# Simulación de una trayectoria
# =============================================================================
def simulate_trajectory(x0, theta, sigma, Tmax):
    x = x0.copy()
    for t in range(Tmax):
        if is_collapsed(x, theta):
            return x, t, 'C'
        if is_recovered(x, theta):
            return x, t, 'R'
        eta = generate_noise(sigma)
        x = hard_nonlinear_dynamics(x, theta, eta)
    return x, Tmax, None

# =============================================================================
# Estimación de committor y MFPT para un punto dado
# =============================================================================
def estimate_q_mfpt(x0, theta, sigma, Tmax, R):
    hits_R = 0
    times_C = []
    for _ in range(R):
        _, t, absorcion = simulate_trajectory(x0, theta, sigma, Tmax)
        if absorcion == 'R':
            hits_R += 1
        elif absorcion == 'C':
            times_C.append(t)
    q_hat = hits_R / R
    mfpt_hat = np.mean(times_C) if times_C else np.nan
    return q_hat, mfpt_hat

# =============================================================================
# Distribución estacionaria de variables rápidas (para proyección)
# =============================================================================
def stationary_fast_given_slow(B, M, theta, sigma, n_samples=1000, burnin=500):
    samples = []
    fast = np.random.uniform(0, 1, size=4)
    for i in range(n_samples + burnin):
        x = np.array([B, M, fast[0], fast[1], fast[2], fast[3]])
        eta = generate_noise(sigma)
        x_new = hard_nonlinear_dynamics(x, theta, eta)
        fast = x_new[2:]
        if i >= burnin:
            samples.append(fast.copy())
    return np.array(samples)

# =============================================================================
# Estimación en grid (B,M) con proyección
# =============================================================================
def estimate_on_grid(B_grid, M_grid, theta, sigma, Tmax, R, fast_samples=1000):
    nB = len(B_grid)
    nM = len(M_grid)
    Q = np.full((nB, nM), np.nan)
    MFPT = np.full((nB, nM), np.nan)
    for i, B in enumerate(tqdm(B_grid, desc="B grid")):
        for j, M in enumerate(M_grid):
            fast_samples_arr = stationary_fast_given_slow(B, M, theta, sigma, n_samples=fast_samples)
            q_vals = []
            mfpt_vals = []
            for fast in fast_samples_arr:
                x0 = np.array([B, M, fast[0], fast[1], fast[2], fast[3]])
                q, mfpt = estimate_q_mfpt(x0, theta, sigma, Tmax, R)
                q_vals.append(q)
                mfpt_vals.append(mfpt)
            Q[i, j] = np.nanmean(q_vals)
            MFPT[i, j] = np.nanmean(mfpt_vals)
    return Q, MFPT

# =============================================================================
# Bootstrap para bandas de confianza
# =============================================================================
def bootstrap_bands(B_grid, M_grid, theta, sigma, Tmax, R,
                    fast_samples=1000, bootstrap_reps=1000, alpha=0.05):
    _, MFPT_hat = estimate_on_grid(B_grid, M_grid, theta, sigma, Tmax, R, fast_samples)
    MFPT_boot = []
    for _ in tqdm(range(bootstrap_reps), desc="Bootstrap"):
        _, MFPT_b = estimate_on_grid(B_grid, M_grid, theta, sigma, Tmax, R, fast_samples)
        MFPT_boot.append(MFPT_b)
    MFPT_boot = np.array(MFPT_boot)
    lower = np.percentile(MFPT_boot, 100*alpha/2, axis=0)
    upper = np.percentile(MFPT_boot, 100*(1-alpha/2), axis=0)
    return MFPT_hat, lower, upper

# =============================================================================
# Validación de proyección
# =============================================================================
def validate_projection(B_test, M_test, theta, sigma, Tmax, R, fast_samples=1000):
    errors = []
    for B, M in zip(B_test, M_test):
        fast_samples_arr = stationary_fast_given_slow(B, M, theta, sigma, n_samples=fast_samples)
        mfpt_2d_vals = []
        for fast in fast_samples_arr:
            x0 = np.array([B, M, fast[0], fast[1], fast[2], fast[3]])
            _, mfpt = estimate_q_mfpt(x0, theta, sigma, Tmax, R)
            mfpt_2d_vals.append(mfpt)
        mfpt_2d = np.nanmean(mfpt_2d_vals)

        mfpt_6d_vals = []
        for _ in range(fast_samples):
            fast = np.random.uniform(0, 1, size=4)
            x0 = np.array([B, M, fast[0], fast[1], fast[2], fast[3]])
            _, mfpt = estimate_q_mfpt(x0, theta, sigma, Tmax, R)
            mfpt_6d_vals.append(mfpt)
        mfpt_6d = np.nanmean(mfpt_6d_vals)

        if not np.isnan(mfpt_6d) and mfpt_6d > 10:
            err = abs(mfpt_2d - mfpt_6d) / mfpt_6d
            errors.append(err)
            print(f"B={B:.2f}, M={M:.2f}: 2D={mfpt_2d:.1f}, 6D={mfpt_6d:.1f}, error={err:.2%}")
        else:
            print(f"B={B:.2f}, M={M:.2f}: MFPT pequeño o NaN, omitido")
    if errors:
        print(f"\nError medio: {np.mean(errors):.2%}, máximo: {np.max(errors):.2%}")
    return errors

# =============================================================================
# Visualización
# =============================================================================
def plot_mfpt(B_grid, M_grid, MFPT, bands_lower=None, title="MFPT on slow manifold"):
    B_mesh, M_mesh = np.meshgrid(B_grid, M_grid, indexing='ij')
    plt.figure(figsize=(8,6))
    contour = plt.contourf(B_mesh, M_mesh, MFPT, levels=20, cmap='viridis')
    plt.colorbar(contour, label='MFPT (log scale)')
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
    print("FOMV: Simulación del modelo HARD-nonlinear")
    print("="*60)

    B_grid = np.linspace(sim_params['B_range'][0], sim_params['B_range'][1],
                         sim_params['Bgrid'])
    M_grid = np.linspace(sim_params['M_range'][0], sim_params['M_range'][1],
                         sim_params['Mgird'])

    print("\nEstimando MFPT en grid (esto puede tomar tiempo)...")
    # Para pruebas rápidas, reduce fast_samples a 10-20
    Q, MFPT = estimate_on_grid(B_grid, M_grid, params, sim_params['sigma'],
                                sim_params['Tmax'], sim_params['R'],
                                fast_samples=50)  # Reducir a 10 para prueba rápida

    print("\nCalculando bandas bootstrap (esto puede tomar mucho tiempo)...")
    # Para pruebas rápidas, reduce bootstrap_reps a 10-20
    MFPT_hat, MFPT_lower, MFPT_upper = bootstrap_bands(
        B_grid, M_grid, params, sim_params['sigma'], sim_params['Tmax'],
        sim_params['R'], fast_samples=20, bootstrap_reps=sim_params['bootstrap_reps'],
        alpha=sim_params['alpha']
    )

    plot_mfpt(B_grid, M_grid, MFPT_hat, bands_lower=MFPT_lower,
              title="MFPT con banda de protección (95%)")

    print("\nValidando proyección con algunos puntos 6D...")
    np.random.seed(42)
    B_test = np.random.uniform(0.2, 1.0, 5)
    M_test = np.random.uniform(0.1, 0.6, 5)
    validate_projection(B_test, M_test, params, sim_params['sigma'],
                        sim_params['Tmax'], sim_params['R'], fast_samples=20)

    print("\nSimulación completada.")
