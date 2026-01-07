import numpy as np
from numpy import pi
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import eigsh

###############################################
# DIY translation-symmetry reduction for L=8  #
# Builds momentum (k) sectors of C_L symmetry #
###############################################

# ---------- Bit utilities ----------

def bit_at(x, i):
    return (x >> i) & 1

def flip_bits(x, i, j):
    mask = (1 << i) | (1 << j)
    return x ^ mask

def rotate_bits_left(x, L, s=1):
    """Cyclic left shift by s on L-bit integer x (site 0 -> 1)."""
    s %= L
    if s == 0:
        return x
    upper = (x << s) & ((1 << L) - 1)
    lower = x >> (L - s)
    return upper | lower

# ---------- Translation group orbits ----------

def orbit_states(b, L):
    """Return unique orbit of basis state b under translations T^r, r=0..L-1.
    Also returns the minimal period R (stabilizer size divisor) s.t. T^R|b>=|b|.
    """
    seen = {}
    states = []
    cur = b
    for r in range(L):
        if cur in seen:
            break
        seen[cur] = r
        states.append(cur)
        cur = rotate_bits_left(cur, L, 1)
    R = len(states)
    return states, R

# ---------- Build momentum basis vectors ----------

def allowed_momenta_for_orbit(R, L):
    """Return list of m in {0..L-1} s.t. e^{ik R} = 1 with k = 2π m / L."""
    return [m for m in range(L) if (m * R) % L == 0]


def momentum_vector_from_orbit(orbit, R, L, m):
    """Construct normalized momentum state |phi_{orbit,m}> = 1/sqrt(R) sum_{t=0}^{R-1} e^{-i k t} |b_t>.
    Returns dense vector of size 2^L (complex128)."""
    dim = 1 << L
    k = 2 * np.pi * m / L
    vec = np.zeros(dim, dtype=np.complex128)
    phase_factors = np.exp(-1j * k * np.arange(R)) / np.sqrt(R)
    for t, b_t in enumerate(orbit):
        vec[b_t] = phase_factors[t]
    return vec


def build_translation_momentum_basis(L):
    """Builds dict: k(m) -> list of basis vectors (dense) spanning the momentum sector.
    Each vector is normalized and constructed from one orbit representative.
    Returns (basis_by_m, reps_by_m), where reps_by_m has (b0, R) for provenance."""
    dim = 1 << L
    used = set()
    basis_by_m = {m: [] for m in range(L)}
    reps_by_m = {m: [] for m in range(L)}
    for b in range(dim):
        if b in used:
            continue
        orbit, R = orbit_states(b, L)
        used.update(orbit)
        allowed = allowed_momenta_for_orbit(R, L)
        for m in allowed:
            vec = momentum_vector_from_orbit(orbit, R, L, m)
            basis_by_m[m].append(vec)
            reps_by_m[m].append((orbit[0], R))
    return basis_by_m, reps_by_m

# ---------- Example Hamiltonian: XXZ Heisenberg chain ----------

def heisenberg_xxz_sparse(L, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0):
    """Return sparse CSR Hamiltonian for spin-1/2 XXZ with periodic BC.
    H = sum_i ( Jx Sx_i Sx_{i+1} + Jy Sy_i Sy_{i+1} + Jz Sz_i Sz_{i+1} ) + h sum_i Sz_i
    Using basis |σ_{L-1} ... σ_0>, σ in {0,1} (0=↓, 1=↑). Sz|↑>=+1/2, Sz|↓>=-1/2.
    """
    dim = 1 << L
    H = dok_matrix((dim, dim), dtype=np.complex128)
    for i in range(L):
        j = (i + 1) % L
        for b in range(dim):
            si = 1 if bit_at(b, i) else -1  # 2*Sz eigenvalue at i
            sj = 1 if bit_at(b, j) else -1
            # Sz Sz term
            H[b, b] += (Jz * 0.25) * (si * sj)
            # Transverse (Sx Sx + Sy Sy) = 1/2 (S+ S- + S- S+)
            bi = bit_at(b, i)
            bj = bit_at(b, j)
            if bi != bj:
                bflip = flip_bits(b, i, j)
                # matrix element is 1/2 (Jx+Jy)/2? For spin-1/2: S+_i S-_j + S-_i S+_j contributes 1/2.
                # Full coefficient for XXZ: (Jx+Jy)/4 per flip term (since SxSx+SySy = 1/2(S+S-+S-S+))
                coeff = 0.5 * (Jx + Jy) * 0.5  # = (Jx+Jy)/4
                H[bflip, b] += coeff
        # on-site field
    if h != 0.0:
        for b in range(dim):
            mz = 0.5 * sum(1 if bit_at(b, i) else -1 for i in range(L))
            H[b, b] += h * mz
    return H.tocsr()

# ---------- Project H into momentum sectors ----------

def project_to_momentum_sector(H, basis_vectors):
    """Given H (CSR, dimension D) and a list of dense basis vectors (each length D),
    build B (D x d) with columns = basis vectors, and return H_k = B^† H B (dense Hermitian)."""
    if not basis_vectors:
        return np.zeros((0, 0), dtype=np.complex128)
    B = np.stack(basis_vectors, axis=1)  # D x d
    HB = H.dot(B)                         # D x d
    Hk = B.conj().T @ HB                  # d x d
    # Symmetrize small numerical noise
    Hk = 0.5 * (Hk + Hk.conj().T)
    return Hk

# ---------- Driver: build blocks for L=8 and solve ----------

def build_blocks_and_solve(L=8, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0, k_list=None, nev=4):
    assert L == 8, "This demo is set up for L=8, but functions are general in L."
    H = heisenberg_xxz_sparse(L, Jx, Jy, Jz, h)
    basis_by_m, reps_by_m = build_translation_momentum_basis(L)
    blocks = {}
    eigs = {}
    for m in (k_list if k_list is not None else range(L)):
        B = basis_by_m[m]
        Hk = project_to_momentum_sector(H, B)
        blocks[m] = Hk
        if Hk.shape[0] == 0:
            eigs[m] = np.array([])
        else:
            # For small blocks use dense solver for robustness
            w = np.linalg.eigh(Hk)[0]
            eigs[m] = w[:nev]
    return H, basis_by_m, reps_by_m, blocks, eigs

if __name__ == "__main__":
    L = 8
    Jx = Jy = 1.0
    Jz = 1.0
    h = 0.0
    H, basis_by_m, reps_by_m, blocks, eigs = build_blocks_and_solve(L, Jx, Jy, Jz, h, nev=6)

    print("Block sizes by momentum m (k=2π m/L):")
    for m in range(L):
        print(f"m={m}: dim={len(basis_by_m[m])}")

    print("\nLowest eigenvalues per sector (first 6):")
    for m in range(L):
        k = 2*np.pi*m/L
        print(f"m={m} (k={k:.3f}): {eigs[m][:6]}")

    # Sanity check: total dimension matches 2^L
    total = sum(len(basis_by_m[m]) for m in range(L))
    print(f"\nTotal momentum basis dim: {total} vs full dim {1<<L}")
