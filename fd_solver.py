import numpy as np
from scipy.linalg import eigh
from radial_tools import radial_grid, effective_potential

def finite_difference_hamiltonian(R, K, l, potential="harmonic", Z=1.0):
    """
    Construct the finite-difference Hamiltonian matrix for the
    radial Schrödinger equation.

    Parameters
    ----------
    R : float
        Radial cutoff
    K : int
        Number of grid intervals
    l : int
        Angular momentum quantum number

    Returns
    -------
    H : (K-1, K-1) ndarray
        Finite-difference Hamiltonian matrix
    r : ndarray
        Interior radial grid points
    """

    r, dr = radial_grid(R, K)
    r_int = r[1:K]           # interior points
    N = K - 1

    W = effective_potential(r_int, l, potential=potential, Z=Z)

    H = np.zeros((N, N))

    diag = 1.0 / dr**2 + W
    off  = -0.5 / dr**2

    np.fill_diagonal(H, diag)
    np.fill_diagonal(H[1:], off)
    np.fill_diagonal(H[:, 1:], off)

    return H, r_int

def solve_finite_difference( R, K, l, n_states=5,potential="harmonic",Z=1.0,verbose=True):  
    """
    Solve the radial Schrödinger equation using the finite-difference method.

    Returns the lowest n_states eigenvalues and eigenvectors,
    sorted from smallest to largest.
    """

    H, r = finite_difference_hamiltonian(
        R, K, l,
        potential=potential,
        Z=Z
    )
    # eigh returns eigenvalues in ascending order
    energies, eigenvectors = eigh(H)

    energies = energies[:n_states]
    eigenvectors = eigenvectors[:, :n_states]

    if verbose:
        print("="*60)
        print("Finite-Difference Method")
        print("="*60)
        print(f"Angular momentum l = {l}")
        print(f"R = {R}, K = {K}")
        print("-"*60)
        print("Lowest eigenvalues:")
        for n, E in enumerate(energies):
            print(f"  n = {n}, ε = {E:.8f}")
        print("="*60)

    return energies, eigenvectors, r


