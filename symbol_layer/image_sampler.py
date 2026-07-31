"""
图像采样器：从真实图像（PNG）中读取模块信息，实现 ModuleSampler 接口。

配合 FrameRenderer 使用：FrameRenderer 渲染 → PNG → ImageSampler 读取 → 解码。
也支持从任意来源的 Cimbar 图像中采样（需提供网格偏移和模块大小）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PIL import Image

from .colors import ColorPalette
from .decoder import ModuleSampler
from .grid import Grid, ModuleType
from .shapes import ShapeSet


class ImageSampler(ModuleSampler):
    """
    从 PIL Image 中采样模块信息。

    Args:
        image: PIL Image（RGB 模式）
        grid: 网格定义
        palette: 颜色调色板（用于颜色匹配）
        shape_set: 图形集合（用于图形识别）
        module_size: 每个模块的像素边长
        quiet_zone_size: 静区宽度（模块数）
    """

    def __init__(
        self,
        image: Image.Image,
        grid: Grid,
        palette: ColorPalette,
        shape_set: ShapeSet,
        module_size: int,
        quiet_zone_size: int = 4,
    ) -> None:
        self.image = image.convert("RGB")
        self.grid = grid
        self.palette = palette
        self.shape_set = shape_set
        self.module_size = module_size
        self.quiet_zone_size = quiet_zone_size
        self._pixel_cache = self.image.load()

    def _module_origin(self, col: int, row: int) -> Tuple[int, int]:
        """计算模块在图像中的像素左上角坐标。"""
        px = (col + self.quiet_zone_size) * self.module_size
        py = (row + self.quiet_zone_size) * self.module_size
        return px, py

    def _sample_region_avg(self, px: int, py: int, ms: int) -> Tuple[int, int, int]:
        """采样一个矩形区域的平均 RGB 值。"""
        r_sum = g_sum = b_sum = 0
        count = 0
        for dy in range(ms):
            for dx in range(ms):
                r, g, b = self._pixel_cache[px + dx, py + dy]
                r_sum += r
                g_sum += g
                b_sum += b
                count += 1
        return (r_sum // count, g_sum // count, b_sum // count)

    def _sample_triangle_avg(
        self, px: int, py: int, ms: int,
        region_fn,
    ) -> Tuple[int, int, int]:
        """采样模块内满足 region_fn 的像素的平均 RGB 值。"""
        r_sum = g_sum = b_sum = 0
        count = 0
        for dy in range(ms):
            for dx in range(ms):
                x = (dx + 0.5) / ms
                y = (dy + 0.5) / ms
                if region_fn(x, y):
                    r, g, b = self._pixel_cache[px + dx, py + dy]
                    r_sum += r
                    g_sum += g
                    b_sum += b
                    count += 1
        if count == 0:
            return (0, 0, 0)
        return (r_sum // count, g_sum // count, b_sum // count)

    def sample_auxiliary_module(self, col: int, row: int) -> Tuple[int, float]:
        """采样辅助模块：返回 (黑白值, 置信度)。"""
        px, py = self._module_origin(col, row)
        ms = self.module_size
        avg = self._sample_region_avg(px, py, ms)

        # 计算到黑/白的距离
        dist_black = sum((c - 0) ** 2 for c in avg)
        dist_white = sum((c - 255) ** 2 for c in avg)

        if dist_black <= dist_white:
            val = 0  # 黑
            confidence = 1.0 - dist_black / (dist_black + dist_white + 1)
        else:
            val = 1  # 白
            confidence = 1.0 - dist_white / (dist_black + dist_white + 1)

        return val, max(0.0, min(1.0, confidence))

    def sample_module_colors(self, col: int, row: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int, float]:
        """
        采样数据模块：返回 (color_a_rgb, color_b_rgb, shape_index, confidence)。

        流程：
        1. 对每个图形假设，采样区域 A 和区域 B 的平均颜色
        2. 匹配到调色板颜色编号
        3. 选择颜色匹配误差最小的图形
        """
        px, py = self._module_origin(col, row)
        ms = self.module_size

        best_shape = 0
        best_error = float("inf")
        best_rgb_a = (0, 0, 0)
        best_rgb_b = (0, 0, 0)

        for shape in self.shape_set.shapes:
            rgb_a = self._sample_triangle_avg(px, py, ms, shape.region_a)
            rgb_b = self._sample_triangle_avg(px, py, ms, shape.region_b)

            # 匹配到调色板
            id_a = self._match_color(rgb_a)
            id_b = self._match_color(rgb_b)
            ref_a = self.palette.get_rgb(id_a)
            ref_b = self.palette.get_rgb(id_b)

            error = sum((a - b) ** 2 for a, b in zip(rgb_a, ref_a))
            error += sum((a - b) ** 2 for a, b in zip(rgb_b, ref_b))

            if error < best_error:
                best_error = error
                best_shape = shape.index
                best_rgb_a = rgb_a
                best_rgb_b = rgb_b

        # 置信度：基于匹配误差
        confidence = 1.0 / (1.0 + best_error / 1000.0)

        return best_rgb_a, best_rgb_b, best_shape, max(0.0, min(1.0, confidence))

    def _match_color(self, rgb: Tuple[int, int, int]) -> int:
        """将 RGB 值匹配到最接近的调色板颜色编号。"""
        best_id = 0
        best_dist = float("inf")
        for cid in self.palette.color_ids:
            pr, pg, pb = self.palette.get_rgb(cid)
            dist = (rgb[0] - pr) ** 2 + (rgb[1] - pg) ** 2 + (rgb[2] - pb) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = cid
        return best_id
