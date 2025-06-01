import numpy as np
from scipy.signal import convolve2d  # CPUで軽量に

LAPLACE = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float32)

def step(u, alpha=0.1, dt=0.1):
    return u + dt * alpha * convolve2d(u, LAPLACE, mode="same", boundary="wrap")
