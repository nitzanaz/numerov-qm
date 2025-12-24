import numpy as np
from scipy.linalg import eigh
from radial_tools import radial_grid, effective_potential



def numerov_matrices(R, K, l,potential="harmonic",Z=1.0): 

    r, dr = radial_grid(R, K)
    r_int = r[1:K]
    Np = K - 1

    W = effective_potential(r_int, l, potential=potential, Z=Z)

    H = np.zeros((Np, Np))
    Nmat = np.zeros((Np, Np))

    a = -1.0 / (2 * dr**2)
    b =  1.0 / (dr**2)

    for k in range(Np):
        H[k, k] = b + (10.0 / 12.0) * W[k]
        Nmat[k, k] = 10.0 / 12.0

        if k > 0:
            H[k, k-1] = a + (1.0 / 12.0) * W[k-1]
            Nmat[k, k-1] = 1.0 / 12.0

        if k < Np - 1:
            H[k, k+1] = a + (1.0 / 12.0) * W[k+1]
            Nmat[k, k+1] = 1.0 / 12.0

    return H, Nmat, r_int

def solve_numerov( R, K, l,n_states=5, potential="harmonic", Z=1.0,verbose=True):
    
    H, Nmat, r = numerov_matrices(
        R, K, l,
        potential=potential,
        Z=Z
    )
    energies, eigenvectors = eigh(H, Nmat)

    energies = energies[:n_states]
    eigenvectors = eigenvectors[:, :n_states]

    if verbose:
        print("="*60)
        print("Numerov Method")
        print("="*60)
        print(f"l = {l}, R = {R}, K = {K}")
        print("-"*60)
        for n, E in enumerate(energies):
            print(f"  n = {n}, ε = {E:.8f}")
        print("="*60)

    return energies, eigenvectors, r
