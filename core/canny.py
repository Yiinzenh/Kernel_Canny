import numpy as np
from core.convolution import convolve
from kernels.gaussian import gaussian_kernel
from kernels.basic import sobel_x, sobel_y


def gaussian_smooth(image, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    return convolve(image, kernel)


def gradient(image):
    gx = convolve(image, sobel_x())
    gy = convolve(image, sobel_y())
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    return mag, angle, gx, gy


def non_max_suppression(mag, angle):
    h, w = mag.shape
    suppressed = np.zeros((h, w), dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = mag[i, j - 1], mag[i, j + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = mag[i - 1, j], mag[i + 1, j]
            else:
                n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]

            if mag[i, j] >= n1 and mag[i, j] >= n2:
                suppressed[i, j] = mag[i, j]
    return suppressed


def double_threshold(suppressed, low, high):
    h, w = suppressed.shape
    strong = suppressed >= high
    weak = (suppressed >= low) & (suppressed < high)
    return strong, weak


def hysteresis(strong, weak):
    h, w = strong.shape
    edges = strong.copy()
    changed = True
    while changed:
        changed = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak[i, j] and not edges[i, j]:
                    if np.any(strong[i - 1:i + 2, j - 1:j + 2]):
                        edges[i, j] = True
                        strong[i, j] = True
                        changed = True
    return edges


def canny(image, gauss_size=5, gauss_sigma=1.0, low=50, high=150):
    smoothed = gaussian_smooth(image, gauss_size, gauss_sigma)
    mag, angle, gx, gy = gradient(smoothed)
    suppressed = non_max_suppression(mag, angle)
    strong, weak = double_threshold(suppressed, low, high)
    edges = hysteresis(strong, weak)
    return {
        "smoothed": smoothed,
        "gradient_mag": mag,
        "gradient_angle": angle,
        "suppressed": suppressed,
        "edges": edges.astype(np.uint8) * 255,
    }
