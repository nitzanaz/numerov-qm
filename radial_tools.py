import numpy as np

def radial_grid(R, K):
    r = np.linspace(0, R, K + 1)
    dr = r[1] - r[0]
    return r, dr


def effective_potential(r, l, potential="harmonic", Z=1.0):
    if potential == "harmonic":
        return 0.5 * r**2 + 0.5 * l * (l + 1) / r**2

    elif potential == "coulomb":
        return -Z / r + 0.5 * l * (l + 1) / r**2

    else:
        raise ValueError("Unknown potential")


def normalize_trapezoidal(u, dr):
    u_full = np.zeros(len(u) + 2)
    u_full[1:-1] = u

    norm2 = np.sum(
        0.5 * dr * (u_full[:-1]**2 + u_full[1:]**2)
    )
    return u / np.sqrt(norm2)


def check_normalization(u, dr):
    u_full = np.zeros(len(u) + 2)
    u_full[1:-1] = u

    return np.sum(
        0.5 * dr * (u_full[:-1]**2 + u_full[1:]**2)
    )
