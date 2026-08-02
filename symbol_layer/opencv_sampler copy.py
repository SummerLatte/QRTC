"""
基于 OpenCV QRCodeDetector 的图像采样器（三阶段精确对齐版）。

流程：
1. _align_rough(): QRCodeDetector.detect() 粗对齐，把图像大致摆正
2. _find_finder_centers(): 在粗对齐图中精确定位 3 个 Finder Pattern 中心（亚像素）
3. _compute_transform(): 用 3 个精确 Finder 中心计算仿射变换矩阵
4. _warp_final(): 最终透视/仿射变换到标准网格
5. 在校正后的图像上采样模块颜色和图形

每一步都输出调试图像，标注关键信息（点、线、框）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .colors import ColorPalette
from .decoder import ModuleSampler
from .grid import Grid
from .shapes import ShapeSet


class OpenCVSampler(ModuleSampler):
    """
    基于 OpenCV QRCodeDetector 的自动定位采样器（三阶段精确对齐）。

    Stage 1: detect() 粗对齐
    Stage 2: Finder Pattern 中心精确定位（质心法，亚像素）
    Stage 3: 仿射变换（3 点，不依赖不准确的 BR）
    Stage 4: 最终变换 + 采样
    """

    # Finder Pattern 中心在模块坐标系中的位置（7×7 Finder，中心在 3.5, 3.5）
    _FINDER_CENTER_MODULES = {
        "tl": (3.5, 3.5),
        "tr": None,  # (N-3.5, 3.5) — 需要运行时计算
        "bl": None,  # (3.5, N-3.5) — 需要运行时计算
    }

    def __init__(
        self,
        image: Image.Image,
        grid: Grid,
        palette: ColorPalette,
        shape_set: ShapeSet,
        debug: bool = False,
    ) -> None:
        self.grid = grid
        self.palette = palette
        self.shape_set = shape_set
        self._debug = debug
        self._debug_images: Dict[str, np.ndarray] = {}

        # 转换为 OpenCV 格式
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        self._cv_img = cv_img

        # 状态
        self._warped = None
        self._module_size = 0
        self._quiet_zone = 4
        self._shape_masks: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self._palette_arr: Optional[np.ndarray] = None

        # 运行时计算 Finder 中心模块坐标
        N = grid.N
        self._finder_centers_modules = {
            "tl": (3.5, 3.5),
            "tr": (N - 3.5, 3.5),
            "bl": (3.5, N - 3.5),
            "br": (N - 3.5, N - 3.5),
        }

        # 三阶段对齐
        self._rough_corners: Optional[np.ndarray] = None
        self._rough_warped: Optional[np.ndarray] = None
        self._rough_ms: float = 0.0
        self._finder_centers_px: Dict[str, np.ndarray] = {}
        self._transform_matrix: Optional[np.ndarray] = None

        self._align_rough()
        self._find_finder_centers()
        self._compute_transform()
        self._warp_final()
        self._precompute()

    # ------------------------------------------------------------------
    # Stage 1: 粗略估计模块大小
    # ------------------------------------------------------------------

    def _align_rough(self) -> None:
        """Stage 1: 用 QRCodeDetector.detect() 检测四角，估计模块大小。

        QRTC 数据模块是三角形（非实心方块），OpenCV detect() 无法直接识别。
        对图像做高斯模糊预处理，使三角形模块近似为均匀色块，再检测。
        不再做粗对齐变换，直接在原始图像上做模板匹配。
        """
        detector = cv2.QRCodeDetector()

        # 尝试直接检测，失败则用模糊预处理
        retval, points = detector.detect(self._cv_img)

        if not retval or points is None:
            # 模糊预处理
            gray = cv2.cvtColor(self._cv_img, cv2.COLOR_BGR2GRAY)
            for blur_r in [6, 8, 10, 12, 14, 16]:
                blurred = cv2.GaussianBlur(gray, (0, 0), blur_r)
                retval, points = detector.detect(blurred)
                if retval and points is not None:
                    break

        # 小图检测失败时，放大后重试
        if not retval or points is None:
            h, w = self._cv_img.shape[:2]
            for scale in [2, 3, 4]:
                scaled = cv2.resize(self._cv_img, (w * scale, h * scale),
                                    interpolation=cv2.INTER_NEAREST)
                retval, points = detector.detect(scaled)
                if retval and points is not None:
                    points = points / scale
                    break
                if not retval or points is None:
                    gray_s = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                    for blur_r in [6, 8, 10, 12, 14, 16]:
                        blurred = cv2.GaussianBlur(gray_s, (0, 0), blur_r)
                        retval, points = detector.detect(blurred)
                        if retval and points is not None:
                            points = points / scale
                            break
                    if retval and points is not None:
                        break

        if not retval or points is None:
            raise ValueError("QRCodeDetector.detect() 未能检测到 Finder Pattern")

        raw_corners = np.float32(points[0])
        # OpenCV detect() 不保证角点顺序，按坐标排序为 TL, TR, BR, BL
        # 先按 y 排序：上两个是 TL/TR，下两个是 BL/BR
        # 再按 x 排序：左是 TL/BL，右是 TR/BR
        sorted_by_y = raw_corners[np.argsort(raw_corners[:, 1])]
        top_two = sorted_by_y[:2]
        bot_two = sorted_by_y[2:]
        tl = top_two[np.argmin(top_two[:, 0])]
        tr = top_two[np.argmax(top_two[:, 0])]
        bl = bot_two[np.argmin(bot_two[:, 0])]
        br = bot_two[np.argmax(bot_two[:, 0])]
        corners = np.float32([tl, tr, br, bl])
        self._rough_corners = corners

        N = self.grid.N

        # 粗略估计模块大小
        dx = np.linalg.norm(tr - tl)
        dy = np.linalg.norm(bl - tl)
        est_ms = ((dx + dy) / 2.0) / N
        self._rough_ms = est_ms

        # 不做粗对齐变换，直接用原始图像
        self._rough_warped = self._cv_img

        if self._debug:
            img1 = self._cv_img.copy()
            for i, pt in enumerate([tl, tr, br, bl]):
                px, py = int(pt[0]), int(pt[1])
                cv2.drawMarker(img1, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img1, ["tl", "tr", "br", "bl"][i], (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self._debug_images["stage1_detect"] = img1

    # ------------------------------------------------------------------
    # Stage 2: 精确定位 Finder Pattern 中心
    # ------------------------------------------------------------------

    def _find_finder_centers(self) -> None:
        """Stage 2: 在原始图像中精确定位 4 个 Finder Pattern 中心（多尺度模板匹配）。

        用 detect() 返回的角点定义 ROI，在每个角区域做模板匹配。
        """
        ms = self._rough_ms
        N = self.grid.N
        H, W = self._cv_img.shape[:2]

        # 用 detect() 返回的角点定义 ROI（角点周围 ±N*ms 区域）
        corners = self._rough_corners
        tl_det, tr_det, br_det, bl_det = corners[0], corners[1], corners[2], corners[3]
        search_r = int(N * ms * 0.5)  # 半个码图大小

        roi_defs = {
            "tl": (max(0, int(tl_det[0]) - search_r), max(0, int(tl_det[1]) - search_r),
                   min(W, int(tl_det[0]) + search_r), min(H, int(tl_det[1]) + search_r)),
            "tr": (max(0, int(tr_det[0]) - search_r), max(0, int(tr_det[1]) - search_r),
                   min(W, int(tr_det[0]) + search_r), min(H, int(tr_det[1]) + search_r)),
            "bl": (max(0, int(bl_det[0]) - search_r), max(0, int(bl_det[1]) - search_r),
                   min(W, int(bl_det[0]) + search_r), min(H, int(bl_det[1]) + search_r)),
            "br": (max(0, int(br_det[0]) - search_r), max(0, int(br_det[1]) - search_r),
                   min(W, int(br_det[0]) + search_r), min(H, int(br_det[1]) + search_r)),
        }

        for name, (x0, y0, x1, y1) in roi_defs.items():
            roi = self._rough_warped[y0:y1, x0:x1]
            center, best_ms = self._locate_finder_center(roi, x0, y0, name, ms)
            self._finder_centers_px[name] = center

        if self._debug:
            img = self._rough_warped.copy()
            # 画 ROI 框
            for name, (x0, y0, x1, y1) in roi_defs.items():
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 128, 0), 1)
                cv2.putText(img, f"{name}_ROI", (x0 + 5, y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
            for name, center in self._finder_centers_px.items():
                cx, cy = int(center[0]), int(center[1])
                cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 3)
                cv2.circle(img, (cx, cy), 8, (0, 255, 0), 2)
                cv2.putText(img, f"{name}", (cx + 12, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            tl = self._finder_centers_px["tl"].astype(int)
            tr = self._finder_centers_px["tr"].astype(int)
            bl = self._finder_centers_px["bl"].astype(int)
            br = self._finder_centers_px["br"].astype(int)
            cv2.line(img, tuple(tl), tuple(tr), (255, 0, 0), 2)
            cv2.line(img, tuple(tl), tuple(bl), (255, 0, 0), 2)
            cv2.line(img, tuple(tr), tuple(br), (255, 0, 0), 2)
            cv2.line(img, tuple(bl), tuple(br), (255, 0, 0), 2)
            self._debug_images["stage2_finder_centers"] = img

    def _locate_finder_center(
        self, roi: np.ndarray, offset_x: int, offset_y: int, name: str, est_ms: int
    ) -> Tuple[np.ndarray, int]:
        """
        在 ROI 中用多尺度模板匹配定位 Finder Pattern 中心。

        对模板大小在 est_ms*0.5 ~ est_ms*2.0 范围内搜索，
        取 TM_SQDIFF_NORMED 最小的匹配作为最佳结果。
        返回 (中心坐标, 最佳模板的模块大小)。
        """
        if roi.size == 0:
            raise ValueError(f"Finder ROI 为空: {name}")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 标准 7×7 Finder Pattern 模板
        finder_template = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 0, 0, 0, 255, 0],
            [0, 255, 0, 0, 0, 255, 0],
            [0, 255, 0, 0, 0, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)

        # 两阶段搜索：先粗步长定位，再精细搜索
        best_val = float("inf")
        best_loc = (0, 0)
        best_ms = est_ms
        best_tmpl_size = 7 * est_ms

        # est_ms 来自 detect() 角点距离，已经很准，只需 ±1 精细搜索
        # 失败时再扩大到 ±3
        est_ms_int = int(est_ms)
        for search_range in [1, 3, max(3, est_ms_int // 2)]:
            ms_min = max(4, est_ms_int - search_range)
            ms_max = max(ms_min + 2, est_ms_int + search_range)
            best_val = float("inf")
            best_loc = (0, 0)
            best_ms = est_ms
            best_tmpl_size = 7 * est_ms
            for try_ms in range(ms_min, ms_max + 1):
                tmpl_size = 7 * try_ms
                if tmpl_size > gray.shape[0] or tmpl_size > gray.shape[1]:
                    break
                template = cv2.resize(finder_template, (tmpl_size, tmpl_size),
                                      interpolation=cv2.INTER_NEAREST)
                result = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
                min_val, _, min_loc, _ = cv2.minMaxLoc(result)
                if min_val < best_val:
                    best_val = min_val
                    best_loc = min_loc
                    best_ms = try_ms
                    best_tmpl_size = tmpl_size
            if best_val < 0.5:  # 找到足够好的匹配
                break

        tx, ty = best_loc

        # 亚像素精度：抛物面拟合
        result = cv2.matchTemplate(
            gray,
            cv2.resize(finder_template, (best_tmpl_size, best_tmpl_size),
                       interpolation=cv2.INTER_NEAREST),
            cv2.TM_SQDIFF_NORMED
        )
        if 1 <= tx < result.shape[1] - 1 and 1 <= ty < result.shape[0] - 1:
            denom_x = result[ty, tx - 1] - 2 * result[ty, tx] + result[ty, tx + 1]
            denom_y = result[ty - 1, tx] - 2 * result[ty, tx] + result[ty + 1, tx]
            dx = 0.5 * (result[ty, tx - 1] - result[ty, tx + 1]) / (denom_x + 1e-12) if abs(denom_x) > 1e-12 else 0.0
            dy = 0.5 * (result[ty - 1, tx] - result[ty + 1, tx]) / (denom_y + 1e-12) if abs(denom_y) > 1e-12 else 0.0
            tx_sub = tx + dx
            ty_sub = ty + dy
        else:
            tx_sub = float(tx)
            ty_sub = float(ty)

        # Finder 中心 = 模板左上角 + 3.5 * best_ms
        cx_roi = tx_sub + 3.5 * best_ms
        cy_roi = ty_sub + 3.5 * best_ms
        cx = cx_roi + offset_x
        cy = cy_roi + offset_y

        if self._debug:
            roi_dbg = roi.copy()
            cv2.rectangle(roi_dbg, (tx, ty), (tx + best_tmpl_size, ty + best_tmpl_size),
                          (0, 255, 255), 2)
            cv2.drawMarker(roi_dbg, (int(cx_roi), int(cy_roi)), (0, 0, 255),
                           cv2.MARKER_CROSS, 15, 2)
            cv2.circle(roi_dbg, (int(cx_roi), int(cy_roi)), 5, (0, 0, 255), 2)
            cv2.putText(roi_dbg, f"{name} ms={best_ms} match={best_val:.4f}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            self._debug_images[f"stage2_roi_{name}"] = roi_dbg

        return np.array([cx, cy], dtype=np.float64), best_ms

    # ------------------------------------------------------------------
    # Stage 3: 计算变换矩阵
    # ------------------------------------------------------------------

    def _compute_transform(self) -> None:
        """Stage 3: 用 4 个精确 Finder 中心计算透视变换矩阵。

        4 点透视变换（8DOF）：4 个精确 Finder 中心。
        BR 角现在有 Finder Pattern，可以直接模板匹配精确定位。
        """
        qz = self._quiet_zone
        N = self.grid.N

        # 目标模块尺寸：每模块至少 16px
        target_px_per_module = 16
        ms = target_px_per_module
        self._module_size = ms

        # 源点：粗对齐图中的 4 个 Finder 中心像素坐标
        src_tl = self._finder_centers_px["tl"]
        src_tr = self._finder_centers_px["tr"]
        src_bl = self._finder_centers_px["bl"]
        src_br = self._finder_centers_px["br"]

        src = np.float32([src_tl, src_tr, src_br, src_bl])

        # 目标点：标准网格中的像素坐标
        fc = self._finder_centers_modules
        dst = np.float32([
            ((fc["tl"][0] + qz) * ms, (fc["tl"][1] + qz) * ms),
            ((fc["tr"][0] + qz) * ms, (fc["tr"][1] + qz) * ms),
            ((fc["br"][0] + qz) * ms, (fc["br"][1] + qz) * ms),
            ((fc["bl"][0] + qz) * ms, (fc["bl"][1] + qz) * ms),
        ])

        # 透视变换（4 点 → 8 自由度）
        self._transform_matrix = cv2.getPerspectiveTransform(src, dst)
        self._transform_is_perspective = True

        if self._debug:
            img = self._rough_warped.copy()
            for name, pt in [("tl", src_tl), ("tr", src_tr), ("br", src_br), ("bl", src_bl)]:
                sx, sy = int(pt[0]), int(pt[1])
                cv2.drawMarker(img, (sx, sy), (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
                cv2.circle(img, (sx, sy), 10, (0, 0, 255), 2)
                cv2.putText(img, f"{name}_src", (sx + 15, sy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            pts = src.astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
            self._debug_images["stage3_transform"] = img

    # ------------------------------------------------------------------
    # Stage 4: 最终变换
    # ------------------------------------------------------------------

    def _warp_final(self) -> None:
        """Stage 4: 用仿射变换矩阵将原图变换到标准网格。"""
        qz = self._quiet_zone
        N = self.grid.N
        ms = self._module_size

        target_w = (N + 2 * qz) * ms
        target_h = (N + 2 * qz) * ms

        self._warped = cv2.warpPerspective(
            self._rough_warped, self._transform_matrix, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        if self._debug:
            img = self._warped.copy()
            # 画网格线（每模块边界）
            for i in range(N + 1):
                pos = (qz + i) * ms
                cv2.line(img, (pos, qz * ms), (pos, (qz + N) * ms),
                         (0, 255, 0) if i % 7 == 0 else (0, 80, 0), 1)
                cv2.line(img, (qz * ms, pos), ((qz + N) * ms, pos),
                         (0, 255, 0) if i % 7 == 0 else (0, 80, 0), 1)
            # 画 Finder Pattern 框
            for name, (mc, mr) in self._finder_centers_modules.items():
                px = int((mc - 3.5 + qz) * ms)
                py = int((mr - 3.5 + qz) * ms)
                cv2.rectangle(img, (px, py), (px + 7 * ms, py + 7 * ms),
                              (255, 0, 0), 2)
            self._debug_images["stage4_final"] = img

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
        ms = self._module_size
        h, w = self._warped.shape[:2]
        x0, y0 = max(0, px), max(0, py)
        x1, y1 = min(w, px + ms), min(h, py + ms)
        if x1 <= x0 or y1 <= y0:
            return (255, 255, 255), (255, 255, 255), 0, 0.0

        region = self._warped[y0:y1, x0:x1]
        rh, rw = region.shape[:2]

        best_shape = 0
        best_error = float("inf")
        best_rgb_a = (0, 0, 0)
        best_rgb_b = (0, 0, 0)

        pal = self._palette_arr

        for shape_idx, (mask_a, mask_b) in enumerate(self._shape_masks):
            sub_a = mask_a[:rh, :rw]
            sub_b = mask_b[:rh, :rw]

            avg_a = cv2.mean(region, mask=sub_a)[:3]
            avg_b = cv2.mean(region, mask=sub_b)[:3]
            rgb_a = (int(avg_a[2]), int(avg_a[1]), int(avg_a[0]))
            rgb_b = (int(avg_b[2]), int(avg_b[1]), int(avg_b[0]))

            # 批量颜色匹配：A 和 B 一起算
            sample = np.array([rgb_a, rgb_b], dtype=np.float32)
            diffs = pal[None, :, :] - sample[:, None, :]
            dists = (diffs ** 2).sum(axis=2)
            id_a, id_b = int(dists[0].argmin()), int(dists[1].argmin())

            ref_a = pal[id_a]
            ref_b = pal[id_b]

            error = (rgb_a[0]-ref_a[0])**2 + (rgb_a[1]-ref_a[1])**2 + (rgb_a[2]-ref_a[2])**2
            error += (rgb_b[0]-ref_b[0])**2 + (rgb_b[1]-ref_b[1])**2 + (rgb_b[2]-ref_b[2])**2

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
