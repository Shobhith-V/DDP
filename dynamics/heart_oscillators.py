"""NumPy heart oscillator simulation (for pre-training, no gradients needed)."""

import numpy as np

# Heart rate ~1-1.5 Hz: omega in rad/s = 2*pi*f_hz


def simulate_coupled_oscillators_numpy(
    T: float = 10.0,
    dt: float = 0.01,
    alpha: float = 1.0,
    omega1_hz: float = 1.0,
    omega2_hz: float = 1.2,
    A_init: float = 0.0001,
    theta_init: float = 3.14159,
    n: float = 1.0,
    modulation: np.ndarray | None = None,
) -> np.ndarray:
    """
    Two coupled Hopf oscillators (polar form).
    dr/dt = alpha*r - r^3 + coupling_r + modulation
    dphi/dt = omega + coupling_phi
    Output: (x1,y1,x2,y2) = (r1*cos(phi1), r1*sin(phi1), ...)
    """
    omega1 = 2 * np.pi * omega1_hz
    omega2 = 2 * np.pi * omega2_hz

    N = int(T / dt)
    r1, r2, phi1, phi2 = 1.0, 1.0, 0.0, 0.0
    A12, A21 = A_init, A_init
    theta12, theta21 = theta_init, theta_init

    R1, R2, Phi1, Phi2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        R1[i], R2[i], Phi1[i], Phi2[i] = r1, r2, phi1, phi2

        phase_diff_12 = theta12 + n * (phi2 - phi1)
        phase_diff_21 = theta21 + n * (phi1 - phi2)

        coupling12 = A12 * r2 * np.cos(phase_diff_12)
        coupling21 = A21 * r1 * np.cos(phase_diff_21)

        dr1 = alpha * r1 - r1**3 + coupling12
        dr2 = alpha * r2 - r2**3 + coupling21
        if modulation is not None and i < len(modulation):
            dr1 += 0.1 * modulation[i, 0]
            dr2 += 0.1 * modulation[i, 1]

        dphi1 = omega1 + A12 * r2 / (r1 + 1e-8) * np.sin(phase_diff_12)
        dphi2 = omega2 + A21 * r1 / (r2 + 1e-8) * np.sin(phase_diff_21)

        r1 = np.clip(r1 + dr1 * dt, 0.01, 2.0)
        r2 = np.clip(r2 + dr2 * dt, 0.01, 2.0)
        phi1 += dphi1 * dt
        phi2 += dphi2 * dt

    return np.stack(
        (R1 * np.cos(Phi1), R1 * np.sin(Phi1), R2 * np.cos(Phi2), R2 * np.sin(Phi2)),
        axis=1,
    )
