"""
符号层解码器：帧图像 → 符号块 + 置信度。

解码方向：
1. 从图像中检测 Finder Pattern、透视校正、推算网格等级
2. 读取 Format Info → 获取颜色档/图形档
3. 按等级参数采样每个数据模块
4. 解码每个数据模块 → 符号值 + 置信度
5. 按扫描顺序拼接符号块

注意：Finder 检测、透视校正、像素级采样依赖外部图像处理库（如 OpenCV）。
本模块定义抽象接口，具体实现由子类或回调提供。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .colors import ColorPalette, ColorPair
from .format_info import FormatInfo
from .grid import Grid, ModuleType
from .module import ModuleCategory
from .shapes import ShapeSet
from .symbol import SymbolSpace


@dataclass
class DecodedSymbol:
    """解码后的单个符号：符号值 + 置信度。"""
    value: int
    confidence: float  # [0, 1]，低置信度将被标记为 erasure

    @property
    def is_erasure(self) -> bool:
        """置信度低于阈值则视为 erasure。"""
        return self.confidence < 0.5


@dataclass
class DecodedFrame:
    """解码后的帧：符号块 + 置信度 + 元信息。"""
    symbol_block: List[int]           # S 个符号值
    confidences: List[float]          # S 个置信度
    erasure_flags: List[bool]         # S 个 erasure 标记
    grid: Grid
    palette: ColorPalette
    shape_set: ShapeSet
    format_info: FormatInfo

    @property
    def S(self) -> int:
        return self.grid.S

    @property
    def M(self) -> int:
        return len(self.palette.pairs) * self.shape_set.n_shapes


# ---- 抽象接口：图像采样器 ----

class ModuleSampler(ABC):
    """
    模块采样器抽象接口。

    负责从实际图像中读取单个模块的颜色信息。
    具体实现需要结合 OpenCV 等图像处理库。
    """

    @abstractmethod
    def sample_module_colors(self, col: int, row: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int, float]:
        """
        采样指定数据模块的两种颜色 RGB 值 + 图形编号 + 置信度。

        Returns:
            (color_a_rgb, color_b_rgb, shape_index, confidence)
            color_a: 区域 A 的平均颜色
            color_b: 区域 B 的平均颜色
            shape_index: 图形编号（由 sampler 的图像分析确定）
            confidence: 识别置信度 [0, 1]
        """
        ...

    @abstractmethod
    def sample_auxiliary_module(self, col: int, row: int) -> Tuple[int, float]:
        """
        采样辅助模块的黑白值 + 置信度。

        Returns:
            (value, confidence)  value: 0=黑, 1=白
        """
        ...


class SymbolDecoder:
    """
    符号层解码器。

    接收采样器接口，从图像中解码符号块。
    Finder 检测和等级推算由外部完成，传入 Grid。
    Format Info 从采样器读取并解码。
    """

    def __init__(self, sampler: ModuleSampler, grid: Grid) -> None:
        self.sampler = sampler
        self.grid = grid

    def decode(self) -> DecodedFrame:
        """
        完整解码流程：
        1. 读取 Format Info → 确定颜色档/图形档
        2. 构建 SymbolSpace
        3. 逐个解码数据模块 → 符号值 + 置信度
        4. 按扫描顺序拼接符号块
        """
        # 1. 读取 Format Info
        fmt, fmt_err = self._read_format_info()
        if fmt_err < 0:
            raise ValueError("Format Info 解码失败，多位错误不可纠正")

        # 2. 构建调色板和图形集合
        palette = ColorPalette.from_code(fmt.color_level_code)
        shape_set = ShapeSet.from_code(fmt.shape_level_code)
        symbol_space = SymbolSpace(palette, shape_set)

        # 3. 解码数据模块
        data_coords = self.grid.data_scan_order
        symbol_block: List[int] = []
        confidences: List[float] = []
        erasure_flags: List[bool] = []

        # 优先使用批量采样（向量化，快 ~2x）
        batch_method = getattr(self.sampler, 'batch_sample_all_modules', None)
        if batch_method is not None:
            shape_arr, ca_arr, cb_arr, conf_arr = batch_method(data_coords)
            for i in range(len(data_coords)):
                i_a, i_b = int(ca_arr[i]), int(cb_arr[i])
                if i_a == i_b:
                    symbol_block.append(0)
                    confidences.append(0.0)
                    erasure_flags.append(True)
                    continue
                shape_idx = int(shape_arr[i])
                # batch 已确保 ca <= cb，无需 flip
                pair = ColorPair(i_a, i_b)
                pair_index = palette.get_pair_index(pair)
                symbol_val = symbol_space.encode(pair_index, shape_idx)
                conf = float(conf_arr[i])
                symbol_block.append(symbol_val)
                confidences.append(conf)
                erasure_flags.append(conf < 0.5)
        else:
            for coord in data_coords:
                symbol_val, conf = self._decode_data_module(
                    coord.col, coord.row, palette, symbol_space
                )
                symbol_block.append(symbol_val)
                confidences.append(conf)
                erasure_flags.append(conf < 0.5)

        return DecodedFrame(
            symbol_block=symbol_block,
            confidences=confidences,
            erasure_flags=erasure_flags,
            grid=self.grid,
            palette=palette,
            shape_set=shape_set,
            format_info=fmt,
        )

    def _read_format_info(self) -> Tuple[FormatInfo, int]:
        """读取两份 Format Info，取纠错结果更好的那份。"""
        positions = self.grid.format_info_positions
        results: List[Tuple[FormatInfo, int]] = []

        for coords in positions:
            bits: List[int] = []
            for c, r in coords:
                val, conf = self.sampler.sample_auxiliary_module(c, r)
                # val: 0=黑 → bit=1, 1=白 → bit=0
                bits.append(1 - val)
            fmt, err = FormatInfo.from_bit_list(bits)
            results.append((fmt, err))

        # 优先选无错的，否则选错误最少的
        for fmt, err in results:
            if err == 0:
                return fmt, err
        # 都有错误，选 error_count 最小的（-1 表示不可纠正，优先级最低）
        best = max(results, key=lambda x: x[1] if x[1] >= 0 else -999)
        return best

    def _decode_data_module(
        self, col: int, row: int,
        palette: ColorPalette,
        symbol_space: SymbolSpace,
    ) -> Tuple[int, float]:
        """
        解码单个数据模块 → (符号值, 置信度)。

        采样模块颜色 → 匹配最接近的颜色对 → 结合 sampler 的图形识别 → 符号值。
        """
        color_a_rgb, color_b_rgb, shape_index, conf = self.sampler.sample_module_colors(col, row)

        # 匹配颜色对：找到最接近的两个颜色编号
        color_a_id = self._match_color(color_a_rgb, palette)
        color_b_id = self._match_color(color_b_rgb, palette)

        # 确定颜色对编号（i < j）
        i, j = min(color_a_id, color_b_id), max(color_a_id, color_b_id)
        if i == j:
            # 纯色模块，不符合规范，标记低置信度
            return 0, 0.0

        # sampler 返回的 shape_index 是基于区域 A/B 的实际空间分布
        # 如果 color_a_id > color_b_id，说明 sampler 看到的区域 A 实际放的是编号大的颜色
        # 这意味着图形方向需要翻转（A↔B 互换）
        if color_a_id > color_b_id:
            shape_index = self._flip_shape(shape_index, symbol_space.n_shapes)

        # 查找颜色对编号
        pair = ColorPair(i, j)
        pair_index = palette.get_pair_index(pair)

        # 编码符号值
        symbol_val = symbol_space.encode(pair_index, shape_index)
        return symbol_val, conf

    def _match_color(self, rgb: Tuple[int, int, int], palette: ColorPalette) -> int:
        """将 RGB 值匹配到最接近的颜色编号（欧氏距离）。"""
        best_id = 0
        best_dist = float("inf")
        for cid in palette.color_ids:
            pr, pg, pb = palette.get_rgb(cid)
            dist = (rgb[0] - pr) ** 2 + (rgb[1] - pg) ** 2 + (rgb[2] - pb) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = cid
        return best_id

    @staticmethod
    def _flip_shape(shape_index: int, n_shapes: int) -> int:
        """
        当区域 A/B 互换时翻转图形编号。

        4 图形档中，互换 A/B 等效于对角翻转：
        0(◣) ↔ 1(◢), 2(◤) ↔ 3(◥)
        2 图形档中：0(◣) ↔ 1(◢)
        对于其他图形档，子类可覆盖此方法。
        """
        if n_shapes == 2:
            return 1 - shape_index
        elif n_shapes == 4:
            return shape_index ^ 1  # 0↔1, 2↔3
        else:
            return shape_index
