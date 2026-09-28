import os
import numpy as np
import matplotlib.pyplot as plt
from utils.io import normalize


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_row(images, titles, save_path, cmap="gray", figsize=None):
    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(normalize(img) if img.dtype != np.uint8 else img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_grid(images, titles, save_path, rows=2, cols=None, cmap="gray"):
    if cols is None:
        cols = len(images) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(normalize(img) if img.dtype != np.uint8 else img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compare(original, results, titles, save_path, cmap="gray"):
    images = [original] + list(results)
    titles_full = ["Original"] + list(titles)
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles_full):
        ax.imshow(normalize(img) if img.dtype != np.uint8 else img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
