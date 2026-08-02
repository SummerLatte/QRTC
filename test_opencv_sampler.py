"""
OpenCVSampler 测试：生成已知符号 → 渲染 → 采样 → 解码 → 对比 → 保存错误模块调试图
"""

import os
import time
import random
import cv2
import numpy as np
from PIL import Image

from symbol_layer import (
    Grid, ColorPalette, ShapeSet,
    SymbolEncoder, FrameRenderer,
    OpenCVSampler, SymbolDecoder,
)

# ---- 配置 ----
GRID_LEVEL = 1
N_COLORS = 4
N_SHAPES = 4
MODULE_SIZE = 20
QUIET_ZONE = 4
SEED = 42

# ---- 构建依赖对象 ----
grid = Grid(GRID_LEVEL)
palette = ColorPalette(N_COLORS)
shape_set = ShapeSet(N_SHAPES)
encoder = SymbolEncoder(palette, shape_set, grid)
renderer = FrameRenderer(module_size=MODULE_SIZE, quiet_zone_size=QUIET_ZONE)

M = len(palette.pairs) * shape_set.n_shapes
S = grid.S
print(f"Grid: N={grid.N}, S={S}, M={M}")

# ---- 生成已知符号块 ----
rng = random.Random(SEED)
original_block = [rng.randint(0, M - 1) for _ in range(S)]

# ---- 编码 + 渲染 ----
frame = encoder.encode(original_block)
img = renderer.render(frame)
IMAGE_PATH = "debug_output/frame_L1_C4_S4_ms20.png"
os.makedirs("debug_output", exist_ok=True)
img.save(IMAGE_PATH)
print(f"图片已保存: {IMAGE_PATH}  尺寸: {img.size}")

# ---- 性能计时：从读图到解出所有符号 ----
print(f"\n==== 性能计时 ====")

t0 = time.perf_counter()
pil_img = Image.open(IMAGE_PATH).convert("RGB")
t1 = time.perf_counter()
print(f"  加载图像: {(t1-t0)*1000:.1f} ms")

sampler = OpenCVSampler(pil_img, grid, palette, shape_set, debug=True)
t2 = time.perf_counter()
print(f"  OpenCVSampler 初始化 (detect+cornerSubPix+warp): {(t2-t1)*1000:.1f} ms")

decoder = SymbolDecoder(sampler, grid)
decoded = decoder.decode()
t3 = time.perf_counter()
print(f"  SymbolDecoder.decode (采样+解码 {S} 模块): {(t3-t2)*1000:.1f} ms")

print(f"  总耗时 (读图→解出符号): {(t3-t0)*1000:.1f} ms")
print(f"  吞吐量: {S / (t3-t0):.0f} 符号/秒")

print(f"\n  符号数: {len(decoded.symbol_block)}")
print(f"  平均置信度: {sum(decoded.confidences) / len(decoded.confidences):.3f}")
print(f"  erasure 数: {sum(decoded.erasure_flags)}")

# ---- 对比原始 vs 解码 ----
data_coords = grid.data_scan_order
errors = []
for i, coord in enumerate(data_coords):
    orig = original_block[i]
    dec = decoded.symbol_block[i]
    conf = decoded.confidences[i]
    if orig != dec or conf < 0.5:
        errors.append((i, coord.col, coord.row, orig, dec, conf))

print(f"\n==== 错误/低置信度模块: {len(errors)} ====")
for i, col, row, orig, dec, conf in errors[:20]:
    color_a, color_b, shape_idx, _ = sampler.sample_module_colors(col, row)
    print(f"  [{i:3d}] ({col:2d},{row:2d}) orig={orig:3d} dec={dec:3d} conf={conf:.3f} "
          f"A={color_a} B={color_b} shape={shape_idx}")
if len(errors) > 20:
    print(f"  ... 共 {len(errors)} 个，仅显示前 20")

# ---- 保存调试图像 ----
for name, img_cv in sampler._debug_images.items():
    path = os.path.join("debug_output", f"sampler_test_{name}.png")
    cv2.imwrite(path, img_cv)
    print(f"  调试图像已保存: {path}")

# ---- 在 warped 图上标注错误模块 ----
if errors:
    warped = sampler._warped.copy()
    qz = sampler._quiet_zone
    ms = int(round(sampler._module_size))
    for i, col, row, orig, dec, conf in errors:
        px = int(round((col + qz) * ms))
        py = int(round((row + qz) * ms))
        cv2.rectangle(warped, (px, py), (px + ms, py + ms), (0, 0, 255), 1)
    err_path = "debug_output/sampler_test_errors.png"
    cv2.imwrite(err_path, warped)
    print(f"  错误模块图已保存: {err_path}")

print("\n完成！")
