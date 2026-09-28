import os
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATASETS = ["alpha", "beta", "salt", "hw"]
LABELS = {"alpha": "AlphaISP", "beta": "BetaISP", "salt": "椒盐", "hw": "手写字"}
COLORS = {"alpha": "#4C72B0", "beta": "#DD8452", "salt": "#C44E52", "hw": "#55A868"}
OUT = os.path.join("outputs", "charts")

BASIC = ["Sobel_X", "Sobel_Y", "Prewitt_X", "Prewitt_Y", "Laplacian_4", "Laplacian_8",
         "Roberts_X", "Roberts_Y", "Texture_0", "Texture_45", "Texture_90", "Texture_135",
         "MeanDiff_3", "MeanDiff_5", "Gauss_3_0.5", "Gauss_5_1.0", "Gauss_7_1.5",
         "GaussGradX_5_1.0", "GaussGradY_5_1.0", "LoG_5_1.0", "LoG_7_1.5"]
GNAME = ["一阶梯度核", "二阶算子", "高斯平滑核", "纹理核", "高斯偏导核"]
GCOLOR = ["#4C72B0", "#C44E52", "#8C8C8C", "#55A868", "#DD8452"]

SWEEP_LABEL = ["5x5 sigma1.0\n60/120", "5x5 sigma1.0\n75/150", "7x7 sigma2.0\n30/60",
               "7x7 sigma2.0\n40/80", "7x7 sigma2.0\n50/100", "7x7 sigma2.0\n80/160"]
SWEEP_VALUE = [7.602, 2.781, 3.189, 1.204, 0.514, 0.045]


def load():
    data = {}
    for k in DATASETS:
        with open(os.path.join("outputs", k, "results.json"), encoding="utf-8") as f:
            data[k] = json.load(f)
    return data


def pick(scores, name):
    if name in scores:
        return scores[name]
    for n, v in scores.items():
        if n.startswith(name):
            return v
    return None


def mean_all(data, name, metric):
    vs = []
    for k in DATASETS:
        for it in data[k]:
            e = pick(it["all_scores"], name)
            if e:
                vs.append(e[metric])
    return float(np.mean(vs)) if vs else np.nan


def mean_each(data, name, metric):
    out = []
    for k in DATASETS:
        vs = []
        for it in data[k]:
            e = pick(it["all_scores"], name)
            if e:
                vs.append(e[metric])
        out.append(float(np.mean(vs)) if vs else np.nan)
    return out


def group_of(name):
    if name.startswith("GaussGrad"):
        return 4
    if name.startswith("Laplacian") or name.startswith("LoG"):
        return 1
    if name.startswith("Gauss"):
        return 2
    if name.startswith("Texture") or name.startswith("MeanDiff"):
        return 3
    return 0


def save(fig, filename):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_kernel_rank(data):
    pairs = sorted(((n, mean_all(data, n, "score")) for n in BASIC), key=lambda p: p[1])
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [GCOLOR[group_of(n)] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(names, vals, color=colors, height=0.7)
    for y, v in enumerate(vals):
        ax.text(v + 0.03, y, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("四份数据集平均综合得分")
    ax.set_ylabel("卷积核")
    ax.set_title("二十一种基础核的平均综合得分排序")
    ax.grid(axis="x", ls=":", alpha=0.5)
    handles = [Patch(color=GCOLOR[i], label=GNAME[i]) for i in range(5)]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    save(fig, "kernel_rank.png")


def chart_canny_density(data):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    pos = 0
    tick_pos, tick_lab = [], []
    for k in DATASETS:
        for it in data[k]:
            d = it["all_scores"]["Canny"]["density"] * 100
            ax.bar(pos, d, width=0.72, color=COLORS[k])
            ax.text(pos, d + 0.12, f"{d:.2f}", ha="center", fontsize=7)
            tick_pos.append(pos)
            tick_lab.append(it["image"])
            pos += 1
    ax.axhline(5, color="#C44E52", ls="--", lw=1.2)
    ax.axhline(2, color="#55A868", ls="--", lw=1.2)
    ax.set_xticks(tick_pos, tick_lab, rotation=60, fontsize=8)
    ax.set_ylabel("Canny 边缘占比 (%)")
    ax.set_title("Canny 检测器在二十张样本上的边缘占比")
    ax.grid(axis="y", ls=":", alpha=0.5)
    handles = [Patch(color=COLORS[k], label=LABELS[k]) for k in DATASETS]
    handles += [plt.Line2D([], [], color="#C44E52", ls="--"),
                plt.Line2D([], [], color="#55A868", ls="--")]
    ax.legend(handles, [LABELS[k] for k in DATASETS] + ["过检参考线 5%", "合理上限参考线 2%"],
              fontsize=8, ncol=3)
    save(fig, "canny_density.png")


def chart_param_sweep():
    x = np.arange(len(SWEEP_VALUE))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x, SWEEP_VALUE, "o-", color="#4C72B0", lw=2, ms=7, label="实测边缘占比")
    for i, v in enumerate(SWEEP_VALUE):
        ax.annotate(f"{v:.3f}%" if v < 0.1 else f"{v:.2f}%", (i, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    best = 4
    ax.plot([best], [SWEEP_VALUE[best]], "o", ms=14, mfc="none", mec="#C44E52", mew=2.2)
    ax.annotate("选定参数", (best, SWEEP_VALUE[best]), textcoords="offset points",
                xytext=(0, -24), ha="center", fontsize=9, color="#C44E52")
    ax.axhline(5, color="#C44E52", ls="--", lw=1.2, label="过检参考线 5%")
    ax.axhline(1, color="#55A868", ls="--", lw=1.2, label="合理上限参考线 1%")
    ax.set_xticks(x, SWEEP_LABEL, fontsize=9)
    ax.set_xlabel("平滑尺度与双阈值组合")
    ax.set_ylabel("Canny 边缘占比 (%)")
    ax.set_title("椒盐数据集参数扫描结果（salt-01）")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    save(fig, "param_sweep.png")


def chart_heat(data):
    names = ["EdgeGroup_v1", "EdgeGroup_v2", "TextureGroup_v1", "Sobel_XY_Mag",
             "Opt_GaussGrad", "Canny", "Gauss_5_1.0", "Gauss_7_1.5", "LoG_5_1.0", "LoG_7_1.5"]
    m = np.array([mean_each(data, n, "consistency") for n in names])
    lim = np.abs(m).max()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(DATASETS)), [LABELS[k] for k in DATASETS])
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(DATASETS)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("代表方案的梯度一致性指标")
    fig.colorbar(im, ax=ax, label="梯度一致性")
    save(fig, "consistency_heat.png")


def chart_scores(data):
    names = ["EdgeGroup_v1", "EdgeGroup_v2", "Opt_GaussGrad", "Sobel_XY_Mag",
             "Canny", "Sobel_X", "LoG_5_1.0"]
    vals = np.array([mean_each(data, n, "score") for n in names])
    x = np.arange(len(DATASETS))
    w = 0.11
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, n in enumerate(names):
        ax.bar(x + (i - 3) * w, vals[i], width=w, label=n)
    ax.set_xticks(x, [LABELS[k] for k in DATASETS])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("平均综合得分")
    ax.set_title("各数据集代表方案的平均综合得分")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=4)
    save(fig, "method_scores.png")


def chart_best(data):
    c = Counter(it["best_method"] for k in DATASETS for it in data[k])
    items = sorted(c.items(), key=lambda p: p[1])
    names = [p[0] for p in items]
    vals = [p[1] for p in items]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.barh(names, vals, color="#4C72B0", height=0.55)
    ax.bar_label(bars, fontsize=10, padding=3)
    ax.set_xlabel("取得最优的样本数")
    ax.set_title("二十张样本的最优方案归属统计")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="x", ls=":", alpha=0.5)
    save(fig, "best_count.png")


def chart_gain(data):
    a = mean_each(data, "EdgeGroup_v1", "score")
    b = mean_each(data, "Canny", "score")
    gain = [(x - y) / abs(y) * 100 for x, y in zip(a, b)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar([LABELS[k] for k in DATASETS], gain,
                  color=[COLORS[k] for k in DATASETS], width=0.55)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=10, padding=3)
    ax.set_ylabel("得分提升幅度 (%)")
    ax.set_title("EdgeGroup_v1 相对 Canny 的综合得分提升")
    ax.grid(axis="y", ls=":", alpha=0.5)
    save(fig, "gain.png")


def main():
    data = load()
    chart_kernel_rank(data)
    chart_canny_density(data)
    chart_param_sweep()
    chart_heat(data)
    chart_scores(data)
    chart_best(data)
    chart_gain(data)
    print("charts saved to", OUT)


if __name__ == "__main__":
    main()
