import numpy as np
from .statistical_physics import T2beta

def Ising_sus_zz(N, S, temp, J=1.0):
    temp = np.atleast_1d(temp)
    chi_zz_array = np.zeros_like(temp, dtype=np.float128)

    states = np.arange(-S, S+1, 1)
    n_states = len(states)
    S_diag = np.diag(states)

    T_switch = 200.0  # above this, use high-T expansion

    for idx, T in enumerate(temp):
        if T > T_switch:
            # high-T analytic approximation (Curie law)
            chi_zz_array[idx] = N * S*(S+1) / (3 * T)
            continue

        beta = T2beta(T)
        outer = np.outer(states, states)
        max_val = np.max(beta * J * outer)
        T_mat = np.exp(beta * J * outer - max_val)

        eigenvals, eigenvecs = np.linalg.eigh(T_mat)
        idx_sort = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx_sort]
        eigenvecs = eigenvecs[:, idx_sort]
        lambda0 = eigenvals[0]
        v0 = eigenvecs[:, 0] / np.sum(eigenvecs[:, 0])

        S2_avg = np.sum(v0**2 * states**2)

        corr_sum = 0.0
        r = np.arange(1, N)
        for k in range(1, n_states):
            lam_ratio = eigenvals[k] / lambda0
            v_k = eigenvecs[:, k]
            corr_coeff = (v0 @ S_diag @ v_k) * (v_k @ S_diag @ v0)
            corr_sum += np.sum(lam_ratio**r * corr_coeff)

        chi_zz_array[idx] = beta * (N * S2_avg + 2 * corr_sum)

    return chi_zz_array

