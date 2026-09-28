import numpy as np


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_gradient_x(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = -xx / (2 * np.pi * sigma ** 4) * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel


def gaussian_gradient_y(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = -yy / (2 * np.pi * sigma ** 4) * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel


def log_kernel(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx ** 2 + yy ** 2
    kernel = (r2 - 2 * sigma ** 2) / (np.pi * sigma ** 4) * np.exp(-r2 / (2 * sigma ** 2))
    return kernel - kernel.sum() / kernel.size


def gaussian_bank(scales=None):
    if scales is None:
        scales = [(3, 0.5), (5, 1.0), (7, 1.5)]
    return [(gaussian_kernel(s, sg), f"Gauss_{s}x{s}_s{sg}") for s, sg in scales]
