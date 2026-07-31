"""
基于 OpenCV QRCodeDetector 的图像采样器。

利用 cv2.QRCodeDetector.detect() 自动检测 Finder Pattern 并获取四角坐标，
通过透视变换校正几何畸变（仿射、旋转、缩放、镜头畸变等），
然后在校正后的图像上采样模块。

流程：
1. cv2.QRCodeDetector.detect() 检测 QR 码四角（依赖 1:1:3:1:1 Finder Pattern）
2. 估计模块大小（从四角点距离推算）
3. 透视变换校正为标准正方形网格（含 quiet zone）
4. 在校正后的图像上采样模块颜色和图形
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .colors import ColorPalette
from .decoder import ModuleSampler
from .grid import Grid
from .shapes import ShapeSet


class OpenCVSampler(ModuleSampler):
    """
    基于 OpenCV QRCodeDetector 的自动定位采样器。

    自动检测 Finder Pattern（通过 1:1:3:1:1 比例扫描），
    获取 QR 码四角坐标，透视变换校正几何畸变后采样。
    不需要预先知道 module_size 或 quiet_zone_size。
    """

    def __init__(
        self,
        image: Image.Image,
        grid: Grid,
        palette: ColorPalette,
        shape_set: ShapeSet,
    ) -> None:
        self.grid = grid
        self.palette = palette
        self.shape_set = shape_set

        # 转换为 OpenCV 格式
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        self._cv_img = cv_img

        # 自动检测并透视校正
        self._warped = None
        self._module_size = 0
        self._quiet_zone = 4
        self._shape_masks: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self._palette_arr: Optional[np.ndarray] = None
        self._align()
        self._precompute()

    def _align(self) -> None:
        """使用 QRCodeDetector.detect() 检测四角，透视变换校正图像。"""
        detector = cv2.QRCodeDetector()
        retval, points = detector.detect(self._cv_img)

        if not retval or points is None:
            raise ValueError("QRCodeDetector.detect() 未能检测到 Finder Pattern")

        # points shape: (1, 4, 2) → 取 points[0] 得到 4 个角点
        # 顺序: [左上, 右上, 右下, 左下] (clockwise from top-left)
        corners = np.float32(points[0])

        tl, tr, br_detect, bl = corners[0], corners[1], corners[2], corners[3]

        # detect() 返回的 br 在无畸变图像上精度较差（底边偏宽）
        # 用对称推算可修正：br_calc = tl + (tr - tl) + (bl - tl)
        # 但在透视畸变（俯拍）下，br_calc 会破坏透视关系
        # 策略：比较 br_detect 和 br_calc 的差距，差距小用 br_calc（精度更高），
        # 差距大用 br_detect（保持真实透视关系）
        br_calc = tl + (tr - tl) + (bl - tl)
        br_diff = np.linalg.norm(br_detect - br_calc)
        avg_span = (np.linalg.norm(tr - tl) + np.linalg.norm(bl - tl)) / 2.0
        relative_diff = br_diff / max(avg_span, 1.0)

        if relative_diff < 0.02:
            # 无明显透视畸变，用计算 br 提高精度
            br = br_calc
        else:
            # 有透视畸变，用 detect 返回的原始 br
            br = br_detect

        # 估计模块大小
        # 四角点围成的区域 = (N + 2*qz) 模块宽高
        # 水平距离: tl→tr = N + 2*qz - 1 模块（像素跨度）
        # 但 detect() 返回的是 QR 码外边界（含 quiet zone 边缘）
        # 实际上 detect() 返回的是包含 Finder Pattern 的最小四边形
        # Finder Pattern 外边界 = 模块 (0,0) 到 (N-1,N-1)
        # 所以 tl→tr 水平距离 ≈ N-1 模块
        qz = self._quiet_zone
        N = self.grid.N

        dx_top = np.linalg.norm(tr - tl)
        dy_left = np.linalg.norm(bl - tl)
        dx_bottom = np.linalg.norm(br - bl)
        dy_right = np.linalg.norm(br - tr)

        avg_w = (dx_top + dx_bottom) / 2.0
        avg_h = (dy_left + dy_right) / 2.0

        # detect() 返回的四角对应 QR 码 Finder Pattern 外边界
        # tl = 模块(0,0)左上角, tr = 模块(N-1,0)右上角附近
        # 实测跨度 ≈ N 模块（含 Finder Pattern 完整外边框）
        est_ms = ((avg_w + avg_h) / 2.0) / N

        # 使用放大的整数模块大小提高采样精度
        # 确保每个模块至少 4 像素，避免大网格下取整误差累积
        target_px_per_module = 4
        scale_factor = max(1, int(np.ceil(target_px_per_module / est_ms)))
        ms = max(target_px_per_module, int(round(est_ms * scale_factor)))
        self._module_size = ms
        self._ms_scale = scale_factor

        # 透视变换：将检测到的四角映射到标准位置
        # detect() 返回的四角是 QR 码外边界：
        # tl = 模块(0,0)左上角, tr ≈ 模块(N,0)左上角
        # 跨度 ≈ N 模块
        src_points = np.float32([tl, tr, br, bl])
        dst_points = np.float32([
            (qz * ms, qz * ms),
            ((qz + N) * ms, qz * ms),
            ((qz + N) * ms, (qz + N) * ms),
            (qz * ms, (qz + N) * ms),
        ])

        target_w = (N + 2 * qz) * ms
        target_h = (N + 2 * qz) * ms

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self._warped = cv2.warpPerspective(
            self._cv_img, matrix, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    def _module_origin(self, col: int, row: int) -> Tuple[int, int]:
        """计算模块在校正图像中的像素左上角坐标。"""
        px = (col + self._quiet_zone) * self._module_size
        py = (row + self._quiet_zone) * self._module_size
        return px, py

    def _sample_region_avg(self, px: int, py: int, ms: int) -> Tuple[int, int, int]:
        """采样一个矩形区域的平均 BGR → RGB 值。"""
        h, w = self._warped.shape[:2]
        x0 = max(0, px)
        y0 = max(0, py)
        x1 = min(w, px + ms)
        y1 = min(h, py + ms)
        if x1 <= x0 or y1 <= y0:
            return (255, 255, 255)
        region = self._warped[y0:y1, x0:x1]
        avg = cv2.mean(region)[:3]
        return (int(avg[2]), int(avg[1]), int(avg[0]))

    def _precompute(self) -> None:
        """预计算 shape mask 和调色板数组，加速采样。"""
        ms = self._module_size

        # 预计算每个 shape 的 region_a / region_b mask
        coords_x = (np.arange(ms) + 0.5) / ms
        coords_y = (np.arange(ms) + 0.5) / ms
        grid_x, grid_y = np.meshgrid(coords_x, coords_y)

        self._shape_masks = []
        for shape in self.shape_set.shapes:
            mask_a = np.where(shape.region_a(grid_x, grid_y), 255, 0).astype(np.uint8)
            mask_b = np.where(shape.region_b(grid_x, grid_y), 255, 0).astype(np.uint8)
            self._shape_masks.append((mask_a, mask_b))

        # 预计算调色板数组
        ids = self.palette.color_ids
        self._palette_arr = np.array([self.palette.get_rgb(cid) for cid in ids], dtype=np.float32)

    def _sample_masked_avg(self, px: int, py: int, mask: np.ndarray) -> Tuple[int, int, int]:
        """用预计算 mask 采样模块内区域的平均 RGB 值。"""
        h, w = self._warped.shape[:2]
        ms = self._module_size
        x0 = max(0, px)
        y0 = max(0, py)
        x1 = min(w, px + ms)
        y1 = min(h, py + ms)
        if x1 <= x0 or y1 <= y0:
            return (255, 255, 255)

        region = self._warped[y0:y1, x0:x1]
        rh, rw = region.shape[:2]

        # 裁剪 mask 到实际区域大小
        sub_mask = mask[:rh, :rw]
        if sub_mask.sum() == 0:
            return (255, 255, 255)

        avg = cv2.mean(region, mask=sub_mask)[:3]
        return (int(avg[2]), int(avg[1]), int(avg[0]))

    def sample_auxiliary_module(self, col: int, row: int) -> Tuple[int, float]:
        """采样辅助模块：返回 (黑白值, 置信度)。"""
        px, py = self._module_origin(col, row)
        ms = self._module_size
        rgb = self._sample_region_avg(px, py, ms)

        dist_black = sum((c - 0) ** 2 for c in rgb)
        dist_white = sum((c - 255) ** 2 for c in rgb)

        if dist_black <= dist_white:
            val = 0
            confidence = 1.0 - dist_black / (dist_black + dist_white + 1)
        else:
            val = 1
            confidence = 1.0 - dist_white / (dist_black + dist_white + 1)

        return val, max(0.0, min(1.0, confidence))

    def sample_module_colors(self, col: int, row: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int, float]:
        """采样数据模块：返回 (color_a_rgb, color_b_rgb, shape_index, confidence)。"""
        px, py = self._module_origin(col, row)

        best_shape = 0
        best_error = float("inf")
        best_rgb_a = (0, 0, 0)
        best_rgb_b = (0, 0, 0)

        for shape_idx, (mask_a, mask_b) in enumerate(self._shape_masks):
            rgb_a = self._sample_masked_avg(px, py, mask_a)
            rgb_b = self._sample_masked_avg(px, py, mask_b)

            id_a = self._match_color(rgb_a)
            id_b = self._match_color(rgb_b)
            ref_a = self._palette_arr[id_a]
            ref_b = self._palette_arr[id_b]

            error = (rgb_a[0] - ref_a[0])**2 + (rgb_a[1] - ref_a[1])**2 + (rgb_a[2] - ref_a[2])**2
            error += (rgb_b[0] - ref_b[0])**2 + (rgb_b[1] - ref_b[1])**2 + (rgb_b[2] - ref_b[2])**2

            if error < best_error:
                best_error = error
                best_shape = shape_idx
                best_rgb_a = rgb_a
                best_rgb_b = rgb_b

        confidence = 1.0 / (1.0 + best_error / 1000.0)
        return best_rgb_a, best_rgb_b, best_shape, max(0.0, min(1.0, confidence))

    def _match_color(self, rgb: Tuple[int, int, int]) -> int:
        """将 RGB 值匹配到最接近的调色板颜色编号。"""
        diffs = self._palette_arr - np.array(rgb, dtype=np.float32)
        dists = (diffs ** 2).sum(axis=1)
        return int(np.argmin(dists))
