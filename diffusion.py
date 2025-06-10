import numpy as np
from scipy.signal import convolve2d  # CPUで軽量に

LAPLACE = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float32)

def step_diff(u, alpha=0.1, dt=0.1):
    return u + dt * alpha * convolve2d(u, LAPLACE, mode="same", boundary="wrap")

def step_wave(u_curr, u_prev, c=1.0, dt=0.1):
    lap_u = convolve2d(u_curr, LAPLACE, mode="same", boundary="wrap")
    return 2 * u_curr - u_prev + (c * dt) ** 2 * lap_u