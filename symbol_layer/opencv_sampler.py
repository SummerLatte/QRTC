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
    基于 OpenCV QRCodeDetector 的自动定位采样器。

    Stage 1: detect() 检测 4 角，直接用 4 角透视变换
    Stage 2: 在校正图像上采样模块颜色和图形
    """

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

        # 4 角在模块坐标系中的位置
        # detect() 返回 QR 码外框角点，对应模块 (0,0) 左上角到 (N-1,N-1) 右下角
        # 跨度 N 个模块宽度
        N = grid.N
        self._corner_modules = {
            "tl": (0.0, 0.0),
            "tr": (float(N), 0.0),
            "br": (float(N), float(N)),
            "bl": (0.0, float(N)),
        }

        # 对齐状态
        self._rough_corners: Optional[np.ndarray] = None
        self._transform_matrix: Optional[np.ndarray] = None

        self._align_and_transform()
        self._precompute()

    # ------------------------------------------------------------------
    # Stage 1: detect() 检测 4 角 + 透视变换
    # ------------------------------------------------------------------

    def _align_and_transform(self) -> None:
        """用 QRCodeDetector.detect() 检测 4 角，直接透视变换到标准网格。"""
        detector = cv2.QRCodeDetector()

        gray = cv2.cvtColor(self._cv_img, cv2.COLOR_BGR2GRAY)
        retval, points = detector.detect(gray)
        if not retval or points is None:
            raise ValueError("QRCodeDetector.detect() 未能检测到 Finder Pattern")

        print("c points======")
        print(points)

        raw_corners = np.float32(points[0])
        sorted_by_y = raw_corners[np.argsort(raw_corners[:, 1])]
        top_two = sorted_by_y[:2]
        bot_two = sorted_by_y[2:]
        tl = top_two[np.argmin(top_two[:, 0])]
        tr = top_two[np.argmax(top_two[:, 0])]
        bl = bot_two[np.argmin(bot_two[:, 0])]
        br = bot_two[np.argmax(bot_two[:, 0])]

        # detect() 的 BR 角点经常偏移（尤其大网格），用 TL/TR/BL 推算更可靠
        br_est = tr + bl - tl
        # 如果 detect 的 BR 与推算值偏差过大（>2px），用推算值替代
        br_dist = np.linalg.norm(br - br_est)
        if br_dist > 2.0:
            br = br_est

        src = np.float32([tl, tr, br, bl])

        # 亚像素精化：在 detect() 角点附近优化到更高精度
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        src_refined = cv2.cornerSubPix(gray, src.reshape(-1, 1, 2), (5, 5), (-1, -1), criteria)
        src = src_refined.reshape(-1, 2)
        self._rough_corners = src.copy()

        print("c refined======")
        print(src)

        # 透视变换到标准网格
        qz = self._quiet_zone
        N = self.grid.N
        # 用精化后的角点间距计算亚像素级模块大小
        # TL→TR 跨度 = N 个模块宽度
        src_ms = np.linalg.norm(src[1] - src[0]) / N
        ms = src_ms
        self._module_size = ms

        cm = self._corner_modules
        dst = np.float32([
            ((cm["tl"][0] + qz) * ms, (cm["tl"][1] + qz) * ms),
            ((cm["tr"][0] + qz) * ms, (cm["tr"][1] + qz) * ms),
            ((cm["br"][0] + qz) * ms, (cm["br"][1] + qz) * ms),
            ((cm["bl"][0] + qz) * ms, (cm["bl"][1] + qz) * ms),
        ])

        self._transform_matrix = cv2.getPerspectiveTransform(src, dst)

        target_w = int(round((N + 2 * qz) * ms))
        target_h = int(round((N + 2 * qz) * ms))
        self._warped = cv2.warpPerspective(
            self._cv_img, self._transform_matrix, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        if self._debug:
            # detect 结果图：精化后角点用绿色圆点标记
            img1 = self._cv_img.copy()
            for pt in src:
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(img1, (px, py), 1, (0, 255, 0), -1)
            self._debug_images["stage1_detect"] = img1

            # 校正后网格图
            img2 = self._warped.copy()
            for i in range(N + 1):
                pos = int(round((qz + i) * ms))
                x0 = int(round(qz * ms))
                x1 = int(round((qz + N) * ms))
                cv2.line(img2, (pos, x0), (pos, x1),
                         (0, 255, 0) if i % 7 == 0 else (0, 80, 0), 1)
                cv2.line(img2, (x0, pos), (x1, pos),
                         (0, 255, 0) if i % 7 == 0 else (0, 80, 0), 1)
            self._debug_images["stage2_warped"] = img2

    def _module_origin(self, col: int, row: int) -> Tuple[int, int]:
        """计算模块在校正图像中的像素左上角坐标。"""
        px = int(round((col + self._quiet_zone) * self._module_size))
        py = int(round((row + self._quiet_zone) * self._module_size))
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
        """预计算 4 三角形 mask（布尔版 + uint8 版）和调色板数组。"""
        ms = int(round(self._module_size))

        coords_x = (np.arange(ms) + 0.5) / ms
        coords_y = (np.arange(ms) + 0.5) / ms
        grid_x, grid_y = np.meshgrid(coords_x, coords_y)

        # 4 个三角形布尔 mask，用于 numpy 批量均值计算
        self._tri_bool = {
            "top":    (grid_y < grid_x) & (grid_y < 1 - grid_x),
            "bottom": (grid_y > grid_x) & (grid_y > 1 - grid_x),
            "left":   (grid_x < grid_y) & (grid_x < 1 - grid_y),
            "right":  (grid_x > grid_y) & (grid_x > 1 - grid_y),
        }
        # uint8 版用于 cv2.mean 兼容
        self._tri_masks = {
            k: np.where(v, 255, 0).astype(np.uint8) for k, v in self._tri_bool.items()
        }

        # 预计算调色板数组
        ids = self.palette.color_ids
        self._palette_arr = np.array([self.palette.get_rgb(cid) for cid in ids], dtype=np.float32)

    def _sample_masked_avg(self, px: int, py: int, mask: np.ndarray) -> Tuple[int, int, int]:
        """用预计算 mask 采样模块内区域的平均 RGB 值。"""
        h, w = self._warped.shape[:2]
        ms = int(round(self._module_size))
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
        ms = int(round(self._module_size))
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
        """
        采样数据模块：返回 (color_a_rgb, color_b_rgb, shape_index, confidence)。

        算法：
        1. 把模块按两条对角线分成上/下/左/右 4 个三角形
        2. 采样 4 个三角形的平均颜色
        3. 比较两种配对方式，选内聚度最高的：
           - 主对角线：上≈左 vs 下≈右 → shape 0/1
           - 副对角线：上≈右 vs 下≈左 → shape 2/3
        4. 匹配两组颜色到调色板，确定颜色对 (i, j)
        5. 根据哪边是小编号颜色确定具体 shape
        """
        px, py = self._module_origin(col, row)
        ms = int(round(self._module_size))
        h, w = self._warped.shape[:2]
        x0, y0 = max(0, px), max(0, py)
        x1, y1 = min(w, px + ms), min(h, py + ms)
        if x1 <= x0 or y1 <= y0:
            return (255, 255, 255), (255, 255, 255), 0, 0.0

        region = self._warped[y0:y1, x0:x1]
        rh, rw = region.shape[:2]

        # 采样 4 个三角形的平均颜色（cv2.mean 对小区域最快）
        masks = self._tri_masks
        region_bgr = region  # BGR 格式
        white = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        def _tri_avg(key):
            sub = masks[key][:rh, :rw]
            if not cv2.countNonZero(sub):
                return white
            avg = cv2.mean(region_bgr, mask=sub)[:3]
            return np.array([avg[2], avg[1], avg[0]], dtype=np.float32)  # BGR→RGB

        c_top = _tri_avg("top")
        c_bot = _tri_avg("bottom")
        c_left = _tri_avg("left")
        c_right = _tri_avg("right")

        # 两种配对的内聚度（组内距离越小越好）
        d_main = float(np.sum((c_top - c_left) ** 2) + np.sum((c_bot - c_right) ** 2))
        d_anti = float(np.sum((c_top - c_right) ** 2) + np.sum((c_bot - c_left) ** 2))

        pal = self._palette_arr

        if d_main <= d_anti:
            avg_a = (c_top + c_left) * 0.5
            avg_b = (c_bot + c_right) * 0.5
        else:
            avg_a = (c_top + c_right) * 0.5
            avg_b = (c_bot + c_left) * 0.5

        # 匹配两个颜色到调色板
        dists_a = ((pal - avg_a) ** 2).sum(axis=1)
        dists_b = ((pal - avg_b) ** 2).sum(axis=1)
        id_a, id_b = int(dists_a.argmin()), int(dists_b.argmin())
        err_a, err_b = float(dists_a.min()), float(dists_b.min())

        if d_main <= d_anti:
            if id_a <= id_b:
                shape_idx = 0
                rgb_a = tuple(int(v) for v in avg_a)
                rgb_b = tuple(int(v) for v in avg_b)
            else:
                shape_idx = 1
                rgb_a = tuple(int(v) for v in avg_b)
                rgb_b = tuple(int(v) for v in avg_a)
                id_a, id_b = id_b, id_a
                err_a, err_b = err_b, err_a
        else:
            if id_a <= id_b:
                shape_idx = 2
                rgb_a = tuple(int(v) for v in avg_a)
                rgb_b = tuple(int(v) for v in avg_b)
            else:
                shape_idx = 3
                rgb_a = tuple(int(v) for v in avg_b)
                rgb_b = tuple(int(v) for v in avg_a)
                id_a, id_b = id_b, id_a
                err_a, err_b = err_b, err_a

        error = err_a + err_b
        confidence = 1.0 / (1.0 + error / 10000.0)
        return rgb_a, rgb_b, shape_idx, max(0.0, min(1.0, confidence))

    def batch_sample_all_modules(self, data_coords) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        向量化批量采样所有数据模块。

        Returns:
            (shape_indices, color_a_ids, color_b_ids, confidences)
            shape_indices: (S,) int32
            color_a_ids: (S,) int32 - 小编号颜色 ID
            color_b_ids: (S,) int32 - 大编号颜色 ID
            confidences: (S,) float32
        """
        N = self.grid.N
        ms = int(round(self._module_size))
        qz = self._quiet_zone
        warped = self._warped
        pal = self._palette_arr

        # 提取整个网格区域并 reshape 为 (N, ms, N, ms, 3)
        x_start = int(round(qz * ms))
        y_start = int(round(qz * ms))
        data_region = warped[y_start:y_start + N * ms, x_start:x_start + N * ms]
        blocks = data_region.reshape(N, ms, N, ms, 3).astype(np.float32)
        blocks = blocks.transpose(0, 2, 1, 3, 4)  # (N_row, N_col, ms, ms, 3) BGR

        # 4 个三角形布尔 mask
        tri = self._tri_bool
        tri_counts = np.array([tri[k].sum() for k in ("top", "bottom", "left", "right")], dtype=np.float32)

        # 批量计算每个三角形的平均颜色
        def _batch_avg(tmask, tcount):
            # blocks: (N, N, ms, ms, 3), tmask: (ms, ms)
            return (blocks * tmask[None, None, :, :, None]).sum(axis=(2, 3)) / tcount

        c_top = _batch_avg(tri["top"], tri_counts[0])     # (N, N, 3) BGR
        c_bot = _batch_avg(tri["bottom"], tri_counts[1])
        c_left = _batch_avg(tri["left"], tri_counts[2])
        c_right = _batch_avg(tri["right"], tri_counts[3])

        # BGR -> RGB
        c_top = c_top[:, :, [2, 1, 0]]
        c_bot = c_bot[:, :, [2, 1, 0]]
        c_left = c_left[:, :, [2, 1, 0]]
        c_right = c_right[:, :, [2, 1, 0]]

        # 内聚度
        d_main = ((c_top - c_left) ** 2).sum(axis=2) + ((c_bot - c_right) ** 2).sum(axis=2)
        d_anti = ((c_top - c_right) ** 2).sum(axis=2) + ((c_bot - c_left) ** 2).sum(axis=2)
        is_main = d_main <= d_anti  # (N, N)

        avg_a_main = (c_top + c_left) * 0.5
        avg_b_main = (c_bot + c_right) * 0.5
        avg_a_anti = (c_top + c_right) * 0.5
        avg_b_anti = (c_bot + c_left) * 0.5

        avg_a = np.where(is_main[:, :, None], avg_a_main, avg_a_anti)  # (N, N, 3)
        avg_b = np.where(is_main[:, :, None], avg_b_main, avg_b_anti)

        # 调色板匹配
        dists_a = ((pal[None, None, :, :] - avg_a[:, :, None, :]) ** 2).sum(axis=3)  # (N, N, n_colors)
        dists_b = ((pal[None, None, :, :] - avg_b[:, :, None, :]) ** 2).sum(axis=3)
        id_a = dists_a.argmin(axis=2)  # (N, N)
        id_b = dists_b.argmin(axis=2)
        err_a = dists_a.min(axis=2)
        err_b = dists_b.min(axis=2)

        a_le_b = id_a <= id_b
        shape_grid = np.where(is_main & a_le_b, 0,
                     np.where(is_main & ~a_le_b, 1,
                     np.where(~is_main & a_le_b, 2, 3))).astype(np.int32)

        # 确保 color_a 是小编号
        color_a_id = np.minimum(id_a, id_b)
        color_b_id = np.maximum(id_a, id_b)

        error = err_a + err_b
        conf_grid = np.clip(1.0 / (1.0 + error / 10000.0), 0.0, 1.0).astype(np.float32)

        # 按扫描顺序提取结果
        S = len(data_coords)
        shape_arr = np.empty(S, dtype=np.int32)
        ca_arr = np.empty(S, dtype=np.int32)
        cb_arr = np.empty(S, dtype=np.int32)
        conf_arr = np.empty(S, dtype=np.float32)
        for i, coord in enumerate(data_coords):
            shape_arr[i] = shape_grid[coord.row, coord.col]
            ca_arr[i] = color_a_id[coord.row, coord.col]
            cb_arr[i] = color_b_id[coord.row, coord.col]
            conf_arr[i] = conf_grid[coord.row, coord.col]

        return shape_arr, ca_arr, cb_arr, conf_arr

    def _match_color(self, rgb: Tuple[int, int, int]) -> int:
        """将 RGB 值匹配到最接近的调色板颜色编号。"""
        diffs = self._palette_arr - np.array(rgb, dtype=np.float32)
        dists = (diffs ** 2).sum(axis=1)
        return int(np.argmin(dists))
