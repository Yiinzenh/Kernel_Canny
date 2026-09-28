# 卷积核与 Canny 边缘及纹理特征提取实验

卷积计算、核构造、Canny 检测器核心流程（NMS / 双阈值分割）全部基于 NumPy 手动实现；
图像读写使用 Pillow，可视化使用 Matplotlib。禁止调用 OpenCV 等库的封装特征提取 API。

## 环境

- Python 3.11（或 Anaconda `yolo` 环境，已验证可用）
- 依赖：numpy、Pillow、matplotlib

```bash
pip install numpy Pillow matplotlib
# 或（Anaconda yolo 环境已自带以上依赖，无需额外安装）
```

## 数据放置

原始数据集放到 `datasets/` 目录下，实验会自动从中选取样本写入 `samples/<dataset_key>/`，
命名格式为 `<数据集名>-XX.<ext>`，并按数据集类型加载：

| 数据集 key | 样本来源目录（噪声图像） |
|---|---|
| alpha | `datasets/AlphaISP - Denoising Dataset/PNG Data/2DNR Denoising/Noised/` |
| beta  | `datasets/BetaISP - Denoising Dataset/Noisy with Ground Truths/Noised/BMP Data/` |
| salt  | `datasets/Salt and Pepper Noise Dataset/椒盐噪声/Noisy_folder/` |
| hw    | `datasets/手写字噪声/train/` |

每份数据集自动选取 5 张样本。

## 运行

```bash
python main.py
# 或用 Anaconda yolo 环境
D:\Anaconda\envs\yolo\python.exe main.py
```

运行完成后结果写入 `outputs/<dataset_key>/`，包含：

- `basic/` 各基础核（Sobel / Laplacian / 高斯系列 / 方向纹理 / 均值差分）单独特征图
- `canny/` Canny 四步流程（平滑 / 梯度 / NMS / 最终边缘）
- `groups/` 多种核组（EdgeGroup、TextureGroup 等）输出
- `optimized/` 针对数据集调参后的优化单核
- `*_basic_grid.png` / `*_groups.png` / `*_compare.png` 汇总对比图
- `results.json` 每张图的量化评估分数与最优方案

## 数据集级参数

主入口 `main.py` 顶部的 `DATASET_PARAMS` 控制每组数据集的高斯尺寸、σ 及 Canny 双阈值。
已按噪声特性预设：椒盐噪声（salt）使用更大的 σ=2.0 / 更大核尺寸 7×7 / 阈值 50/100；
其余自然图像与手写字数据集使用 σ=1.0 / 5×5 / 40/120。如需调整，修改该字典即可，
后续 Canny、核组、评估会自动使用同一组参数。

## 代码组织

- `main.py` 主入口：样本收集 → 基础核 / Canny / 核组 / 优化单核 → 评估 → 结果写出
- `core/convolution.py` NumPy 手写零填充卷积（双重循环 + 向量化）
- `core/canny.py` 手动 Canny 流程（高斯平滑 → 梯度 → NMS → 双阈值 + 迟滞）
- `core/evaluate.py` 量化评估（梯度一致性、连通性、边缘密度）
- `kernels/gaussian.py` 高斯平滑核 / 一阶偏导 / LoG / 多尺度组
- `kernels/basic.py` Sobel / Prewitt / Laplacian / Roberts
- `kernels/texture.py` 方向纹理核 / 均值差分核 / Gabor-like
- `utils/io.py` 图像读写、灰度转换、百分位裁剪归一化
- `utils/visualize.py` Matplotlib 对比可视化
