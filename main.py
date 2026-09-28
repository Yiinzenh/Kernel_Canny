import os
import json
import numpy as np

from utils.io import load_gray, normalize, save_image, list_images
from utils.visualize import plot_row, plot_grid, plot_compare
from core.convolution import convolve
from core.canny import canny, gaussian_smooth, gradient
from kernels.gaussian import (gaussian_kernel, gaussian_gradient_x,
                             gaussian_gradient_y, log_kernel)
from kernels.basic import (sobel_x, sobel_y, prewitt_x, prewitt_y,
                          laplacian_4, laplacian_8, roberts_x, roberts_y)
from kernels.texture import directional_texture, mean_diff_kernel, gabor_like


SAMPLES = {
    "alpha": "samples/alpha",
    "beta":  "samples/beta",
    "salt":  "samples/salt",
    "hw":    "samples/hw",
}

DATASET_LABEL = {
    "alpha": "AlphaISP自然风景",
    "beta":  "BetaISP自然风景",
    "salt":  "椒盐噪声集",
    "hw":    "手写字噪声集",
}
OUTPUT = "outputs"

DATASET_PARAMS = {
    "alpha": {"gsize": 5, "gsigma": 1.0, "low": 40, "high": 120},
    "beta":  {"gsize": 5, "gsigma": 1.0, "low": 40, "high": 120},
    "salt":  {"gsize": 7, "gsigma": 2.0, "low": 50, "high": 100},
    "hw":    {"gsize": 5, "gsigma": 1.0, "low": 40, "high": 120},
}


def basic_edge_kernels():
    return [
        ("Sobel_X", sobel_x()),
        ("Sobel_Y", sobel_y()),
        ("Prewitt_X", prewitt_x()),
        ("Prewitt_Y", prewitt_y()),
        ("Laplacian_4", laplacian_4()),
        ("Laplacian_8", laplacian_8()),
        ("Roberts_X", roberts_x()),
        ("Roberts_Y", roberts_y()),
    ]


def basic_texture_kernels():
    return [
        ("Texture_0", directional_texture(0)),
        ("Texture_45", directional_texture(45)),
        ("Texture_90", directional_texture(90)),
        ("Texture_135", directional_texture(135)),
        ("MeanDiff_3", mean_diff_kernel(3)),
        ("MeanDiff_5", mean_diff_kernel(5)),
    ]


def gaussian_kernels():
    return [
        ("Gauss_3_0.5", gaussian_kernel(3, 0.5)),
        ("Gauss_5_1.0", gaussian_kernel(5, 1.0)),
        ("Gauss_7_1.5", gaussian_kernel(7, 1.5)),
        ("GaussGradX_5_1.0", gaussian_gradient_x(5, 1.0)),
        ("GaussGradY_5_1.0", gaussian_gradient_y(5, 1.0)),
        ("LoG_5_1.0", log_kernel(5, 1.0)),
        ("LoG_7_1.5", log_kernel(7, 1.5)),
    ]


def normalize_for_chain(arr):
    n = normalize(arr).astype(np.float64)
    return n


def kernel_series(image, kernels):
    result = image.copy()
    for k in kernels:
        result = convolve(result, k)
        result = normalize_for_chain(result)
    return result


def kernel_parallel(image, kernels, weights=None):
    outputs = [convolve(image, k) for k in kernels]
    outputs = [normalize(o).astype(np.float64) for o in outputs]
    if weights is None:
        weights = [1.0 / len(kernels)] * len(kernels)
    fused = sum(w * o for w, o in zip(weights, outputs))
    return normalize(fused)


def edge_kernel_group_v1(image, gsize, gsigma):
    smoothed = convolve(image, gaussian_kernel(gsize, gsigma))
    gx = convolve(smoothed, sobel_x())
    gy = convolve(smoothed, sobel_y())
    return normalize(np.sqrt(gx ** 2 + gy ** 2))


def edge_kernel_group_v2(image, gsize, gsigma):
    fine = convolve(image, gaussian_kernel(3, gsigma * 0.5))
    coarse = convolve(image, gaussian_kernel(gsize, gsigma))
    gx1 = convolve(fine, sobel_x())
    gy1 = convolve(fine, sobel_y())
    gx2 = convolve(coarse, sobel_x())
    gy2 = convolve(coarse, sobel_y())
    mag1 = normalize(np.sqrt(gx1 ** 2 + gy1 ** 2)).astype(np.float64)
    mag2 = normalize(np.sqrt(gx2 ** 2 + gy2 ** 2)).astype(np.float64)
    return normalize(mag1 * 0.6 + mag2 * 0.4)


def texture_kernel_group_v1(image, gsigma):
    smoothed = convolve(image, gaussian_kernel(3, gsigma * 0.5))
    t0 = convolve(smoothed, directional_texture(0))
    t90 = convolve(smoothed, directional_texture(90))
    fused = normalize(t0).astype(np.float64) * 0.5 + normalize(t90).astype(np.float64) * 0.5
    return normalize(fused)


def process_basic_kernels(image, save_dir, img_name):
    results = {}
    all_kernels = basic_edge_kernels() + basic_texture_kernels() + gaussian_kernels()

    for name, k in all_kernels:
        feat = convolve(image, k)
        results[name] = feat
        save_image(feat, os.path.join(save_dir, "basic", f"{img_name}_{name}.png"))

    names = list(results.keys())
    images = [results[n] for n in names]
    plot_grid(images, names,
              os.path.join(save_dir, f"{img_name}_basic_grid.png"),
              rows=3, cols=7)

    return results


def process_canny(image, save_dir, img_name, gsize, gsigma, low, high):
    result = canny(image, gauss_size=gsize, gauss_sigma=gsigma, low=low, high=high)

    save_image(result["smoothed"], os.path.join(save_dir, "canny", f"{img_name}_smoothed.png"))
    save_image(result["gradient_mag"], os.path.join(save_dir, "canny", f"{img_name}_grad_mag.png"))
    save_image(result["suppressed"], os.path.join(save_dir, "canny", f"{img_name}_nms.png"))
    save_image(result["edges"], os.path.join(save_dir, "canny", f"{img_name}_edges.png"))

    plot_row(
        [image, result["smoothed"], result["gradient_mag"], result["suppressed"], result["edges"]],
        ["Original", "Smoothed", "Gradient", "NMS", "Canny Edges"],
        os.path.join(save_dir, f"{img_name}_canny_pipeline.png"),
    )

    return result


def process_kernel_groups(image, save_dir, img_name, gsize, gsigma):
    groups = {
        "EdgeGroup_v1": edge_kernel_group_v1(image, gsize, gsigma),
        "EdgeGroup_v2": edge_kernel_group_v2(image, gsize, gsigma),
        "TextureGroup_v1": texture_kernel_group_v1(image, gsigma),
        "Sobel_XY_Mag": normalize(np.sqrt(convolve(image, sobel_x()) ** 2 + convolve(image, sobel_y()) ** 2)),
        "LoG_Base": convolve(image, log_kernel(gsize, gsigma)),
    }

    for name, feat in groups.items():
        save_image(feat, os.path.join(save_dir, "groups", f"{img_name}_{name}.png"))

    names = list(groups.keys())
    images = list(groups.values())
    plot_row(images, names,
             os.path.join(save_dir, f"{img_name}_groups.png"),
             figsize=(4 * len(names), 4))

    return groups


def process_single_optimized(image, save_dir, img_name, gsize, gsigma):
    fine = gaussian_kernel(3, gsigma * 0.5)
    optimized = {
        "Opt_Sobel_Gauss3": normalize(
            np.sqrt(convolve(convolve(image, fine), sobel_x()) ** 2 +
                    convolve(convolve(image, fine), sobel_y()) ** 2)
        ),
        f"Opt_LoG_{gsize}_{gsigma}": convolve(image, log_kernel(gsize, gsigma)),
        f"Opt_GaussGrad_{gsize}_{gsigma}": normalize(
            np.sqrt(convolve(image, gaussian_gradient_x(gsize, gsigma)) ** 2 +
                    convolve(image, gaussian_gradient_y(gsize, gsigma)) ** 2)
        ),
        f"Opt_Gabor45_{gsize}": convolve(image, gabor_like(45, size=gsize, sigma=gsigma, freq=0.25)),
    }

    for name, feat in optimized.items():
        save_image(feat, os.path.join(save_dir, "optimized", f"{img_name}_{name}.png"))

    names = list(optimized.keys())
    images = list(optimized.values())
    plot_row(images, names,
             os.path.join(save_dir, f"{img_name}_optimized.png"),
             figsize=(4 * len(names), 4))

    return optimized


def connectivity_score(binary):
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    num_components = 0
    total_edge = 0
    for i in range(h):
        for j in range(w):
            if binary[i, j] and not visited[i, j]:
                num_components += 1
                stack = [(i, j)]
                comp_size = 0
                while stack:
                    x, y = stack.pop()
                    if x < 0 or x >= h or y < 0 or y >= w:
                        continue
                    if visited[x, y] or not binary[x, y]:
                        continue
                    visited[x, y] = True
                    comp_size += 1
                    total_edge += 1
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((x + dx, y + dy))
    if num_components == 0 or total_edge == 0:
        return 0.0
    avg_size = total_edge / num_components
    return avg_size / max(h, w)


def gradient_consistency(binary, gradient_mag):
    if binary.sum() == 0:
        return 0.0
    edge_mag = gradient_mag[binary > 0].mean()
    non_edge_mag = gradient_mag[binary == 0].mean() + 1e-10
    return float(np.log(edge_mag / non_edge_mag))


def select_best(candidates, original, gsize, gsigma):
    _, _, gx, gy = gradient(gaussian_smooth(original, gsize, gsigma))
    gradient_mag = np.sqrt(gx ** 2 + gy ** 2)

    best_name = None
    best_score = -1e18
    scores = {}

    for name, feat in candidates.items():
        feat_n = normalize(feat).astype(np.float64)
        unique_vals = np.unique(feat_n)
        is_binary = len(unique_vals) <= 2

        if is_binary:
            binary = (feat_n > 0).astype(np.float64)
        else:
            thr = np.percentile(feat_n, 95)
            binary = (feat_n > thr).astype(np.float64)

        density = float(np.mean(binary))
        conn = connectivity_score(binary)
        consistency = gradient_consistency(binary, gradient_mag)
        density_penalty = abs(density - 0.05) * 3.0
        score = consistency + conn * 0.5 - density_penalty

        scores[name] = {
            "density": density,
            "connectivity": conn,
            "consistency": consistency,
            "score": score,
        }
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, scores


def run_dataset(dataset_key, dataset_name):
    sample_dir = SAMPLES[dataset_key]
    out_dir = os.path.join(OUTPUT, dataset_key)
    images = list_images(sample_dir)
    p = DATASET_PARAMS[dataset_key]
    gsize, gsigma, low, high = p["gsize"], p["gsigma"], p["low"], p["high"]

    annotations = []

    for img_path in images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing {dataset_name}/{img_name} ...")

        image = load_gray(img_path)

        basic = process_basic_kernels(image, out_dir, img_name)
        canny_res = process_canny(image, out_dir, img_name, gsize, gsigma, low, high)
        groups = process_kernel_groups(image, out_dir, img_name, gsize, gsigma)
        optimized = process_single_optimized(image, out_dir, img_name, gsize, gsigma)

        candidates = {}
        candidates.update(basic)
        candidates["Canny"] = canny_res["edges"]
        candidates.update(groups)
        candidates.update(optimized)

        best_name, scores = select_best(candidates, image, gsize, gsigma)

        compare = {
            f"Best: {best_name}": candidates[best_name],
            f"Canny s{gsigma} {low}/{high}": canny_res["edges"],
            "Sobel_X raw": basic["Sobel_X"],
            f"LoG {gsize}x{gsize} s{gsigma}": convolve(image, log_kernel(gsize, gsigma)),
            f"EdgeGroup Gauss{gsize} s{gsigma}": groups["EdgeGroup_v1"],
        }
        plot_compare(
            image,
            list(compare.values()),
            list(compare.keys()),
            os.path.join(out_dir, f"{img_name}_compare.png"),
        )

        annotations.append({
            "image": img_name,
            "params": {"gauss_size": gsize, "gauss_sigma": gsigma,
                       "canny_low": low, "canny_high": high},
            "best_method": best_name,
            "best_scores": scores.get(best_name, {}),
            "all_scores": scores,
        })

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    return annotations


def main():
    for key, name in DATASET_LABEL.items():
        run_dataset(key, name)
    print("All done. Results saved to outputs/")


if __name__ == "__main__":
    main()
