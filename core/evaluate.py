import numpy as np


def edge_clarity(gradient_magnitude):
    return float(np.mean(gradient_magnitude ** 2))


def noise_suppression(feature_map, original):
    if feature_map.shape != original.shape:
        return 0.0
    diff = np.abs(feature_map.astype(np.float64) - original.astype(np.float64))
    return float(np.mean(diff))


def edge_density(edge_map):
    if edge_map.dtype == bool:
        return float(np.mean(edge_map))
    return float(np.mean(edge_map > 0))


def signal_to_noise_ratio(feature_map):
    f = feature_map.astype(np.float64)
    signal = np.mean(f)
    noise = np.std(f)
    if noise < 1e-10:
        return 0.0
    return float(20 * np.log10(abs(signal) / noise + 1e-10))


def evaluate(edge_map, gradient_mag=None, original=None):
    scores = {}
    scores["edge_density"] = edge_density(edge_map)
    if gradient_mag is not None:
        scores["clarity"] = edge_clarity(gradient_mag)
        scores["snr"] = signal_to_noise_ratio(gradient_mag)
    if original is not None:
        scores["noise_suppress"] = noise_suppression(edge_map, original)
    return scores
