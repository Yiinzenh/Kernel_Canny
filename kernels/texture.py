import numpy as np


def mean_diff_kernel(size=3):
    kernel = np.ones((size, size), dtype=np.float64)
    center = size // 2
    kernel[center, center] = -(size * size - 1)
    return kernel


def directional_texture(angle_deg, size=5):
    angle = np.deg2rad(angle_deg)
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    direction = np.cos(angle) * xx + np.sin(angle) * yy
    kernel = np.sin(direction * np.pi / (size / 2))
    return kernel - kernel.mean()


def gabor_like(angle_deg, size=5, sigma=1.0, freq=0.3):
    angle = np.deg2rad(angle_deg)
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    x_prime = xx * np.cos(angle) + yy * np.sin(angle)
    y_prime = -xx * np.sin(angle) + yy * np.cos(angle)
    gauss = np.exp(-(x_prime ** 2 + y_prime ** 2) / (2 * sigma ** 2))
    wave = np.cos(2 * np.pi * freq * x_prime)
    kernel = gauss * wave
    return kernel - kernel.mean()


def texture_bank():
    angles = [0, 45, 90, 135]
    kernels = []
    for a in angles:
        k = directional_texture(a)
        kernels.append((k, f"Texture_dir{a}"))
    k2 = mean_diff_kernel(3)
    kernels.append((k2, "MeanDiff_3x3"))
    k3 = mean_diff_kernel(5)
    kernels.append((k3, "MeanDiff_5x5"))
    return kernels
