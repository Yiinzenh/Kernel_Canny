import numpy as np


def sobel_x():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)


def sobel_y():
    return np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=np.float64)


def prewitt_x():
    return np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]], dtype=np.float64)


def prewitt_y():
    return np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]], dtype=np.float64)


def laplacian_4():
    return np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]], dtype=np.float64)


def laplacian_8():
    return np.array([[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]], dtype=np.float64)


def roberts_x():
    return np.array([[1, 0],
                     [0, -1]], dtype=np.float64)


def roberts_y():
    return np.array([[0, 1],
                     [-1, 0]], dtype=np.float64)
