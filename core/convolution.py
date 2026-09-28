import numpy as np


def convolve2d(image, kernel, pad_value=0):
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    ih, iw = image.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant", constant_values=pad_value)

    out = np.zeros_like(image, dtype=np.float64)
    flipped = kernel[::-1, ::-1]

    for i in range(ih):
        for j in range(iw):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * flipped)

    return out


def convolve2d_vectorized(image, kernel, pad_value=0):
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    ih, iw = image.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant", constant_values=pad_value)

    flipped = kernel[::-1, ::-1]

    col_shape = (ih * iw, kh * kw)
    cols = np.zeros(col_shape, dtype=np.float64)
    idx = 0
    for i in range(ih):
        for j in range(iw):
            cols[idx] = padded[i:i + kh, j:j + kw].ravel()
            idx += 1

    out = cols @ flipped.ravel()
    return out.reshape(ih, iw)


def convolve(image, kernel, pad_value=0):
    return convolve2d_vectorized(image, kernel, pad_value)
