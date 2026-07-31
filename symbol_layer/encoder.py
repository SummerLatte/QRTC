"""
符号层编码器：符号块 → 帧图像描述。

编码方向：
1. 接收符号块（S 个符号值，每个 0 到 M-1）
2. 将符号值映射为颜色对 + 图形
3. 渲染到网格的数据模块位置
4. 填充结构区域（Finder、Separator、Timing、Format Info）
5. 输出 RenderedFrame（完整的模块描述矩阵）

本模块输出的是逻辑描述（每个模块的颜色/图形），不涉及像素级渲染。
像素级渲染由上层（如 OpenCV/PIL）根据模块描述完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .colors import ColorPalette
from .format_info import FormatInfo
from .grid import Grid, ModuleType, FINDER_PATTERN
from .module import ModuleCategory, RenderedModule
from .shapes import ShapeSet
from .symbol import SymbolSpace


@dataclass
class RenderedFrame:
    """
    渲染后的帧描述：完整的 N×N 模块矩阵。

    每个元素是 RenderedModule，描述该模块的颜色和图形。
    上层据此进行像素级渲染。
    """
    grid: Grid
    palette: ColorPalette
    shape_set: ShapeSet
    modules: List[List[RenderedModule]]  # [row][col]

    @property
    def N(self) -> int:
        return self.grid.N

    def get_module(self, col: int, row: int) -> RenderedModule:
        return self.modules[row][col]


class SymbolEncoder:
    """
    符号层编码器。

    将符号块编码为帧的模块描述矩阵。
    """

    def __init__(self, palette: ColorPalette, shape_set: ShapeSet, grid: Grid) -> None:
        self.palette = palette
        self.shape_set = shape_set
        self.grid = grid
        self.symbol_space = SymbolSpace(palette, shape_set)

    @property
    def S(self) -> int:
        """单帧符号容量。"""
        return self.grid.S

    @property
    def M(self) -> int:
        """符号总数。"""
        return self.symbol_space.M

    def encode(self, symbol_block: List[int]) -> RenderedFrame:
        """
        将符号块编码为帧描述。

        Args:
            symbol_block: S 个符号值，每个 [0, M-1]
        Returns:
            RenderedFrame: 完整的模块描述矩阵
        """
        if len(symbol_block) != self.S:
            raise ValueError(f"符号块长度 {len(symbol_block)} != S={self.S}")

        for i, sv in enumerate(symbol_block):
            if not 0 <= sv < self.M:
                raise ValueError(f"符号值越界 @[{i}]: {sv}, 范围 [0, {self.M})")

        # 构建 Format Info
        fmt = FormatInfo.from_codes(self.palette.code, self.shape_set.code)
        fmt_bits = fmt.to_bit_list()  # 14 bits

        # 初始化模块矩阵
        N = self.grid.N
        modules: List[List[RenderedModule]] = [[None] * N for _ in range(N)]

        # 填充结构区域
        self._fill_finder_and_separator(modules)
        self._fill_timing(modules)
        self._fill_format_info(modules, fmt_bits)

        # 填充数据模块
        data_coords = self.grid.data_scan_order
        for i, coord in enumerate(data_coords):
            symbol_val = symbol_block[i]
            color_a, color_b, shape = self.symbol_space.get_colors_for_symbol(symbol_val)
            modules[coord.row][coord.col] = RenderedModule(
                col=coord.col,
                row=coord.row,
                category=ModuleCategory.DATA,
                color_a=color_a,
                color_b=color_b,
                shape_index=shape.index,
            )

        return RenderedFrame(
            grid=self.grid,
            palette=self.palette,
            shape_set=self.shape_set,
            modules=modules,
        )

    def _fill_finder_and_separator(self, modules: List[List[RenderedModule]]) -> None:
        """填充 Finder Pattern 和 Separator。"""
        N = self.grid.N
        finder_positions = [(0, 0), (N - 7, 0), (0, N - 7)]

        for fc, fr in finder_positions:
            for dr in range(7):
                for dc in range(7):
                    c, r = fc + dc, fr + dr
                    val = FINDER_PATTERN[dr][dc]
                    color = 0 if val == 1 else 1  # 1=黑(0), 0=白(1)
                    modules[r][c] = RenderedModule(
                        col=c, row=r,
                        category=ModuleCategory.AUXILIARY,
                        color_a=color, color_b=color,
                        shape_index=None,
                    )

        # Separator: 白色边框
        for fc, fr in finder_positions:
            for dr in range(-1, 8):
                for dc in range(-1, 8):
                    c, r = fc + dc, fr + dr
                    if 0 <= c < N and 0 <= r < N and modules[r][c] is None:
                        mt = self.grid.get_module_type(c, r)
                        if mt == ModuleType.SEPARATOR:
                            modules[r][c] = RenderedModule(
                                col=c, row=r,
                                category=ModuleCategory.AUXILIARY,
                                color_a=1, color_b=1,  # 白
                                shape_index=None,
                            )

    def _fill_timing(self, modules: List[List[RenderedModule]]) -> None:
        """填充 Timing Pattern（交替黑白）。"""
        N = self.grid.N
        # 水平: row=6, col=8 到 col=N-9
        for c in range(8, N - 8):
            if modules[6][c] is not None:
                continue
            mt = self.grid.get_module_type(c, 6)
            if mt == ModuleType.TIMING:
                val = (c - 8) % 2  # col=8 → 0(黑), col=9 → 1(白), ...
                color = 0 if val == 0 else 1
                modules[6][c] = RenderedModule(
                    col=c, row=6,
                    category=ModuleCategory.AUXILIARY,
                    color_a=color, color_b=color,
                    shape_index=None,
                )

        # 垂直: col=6, row=8 到 row=N-9
        for r in range(8, N - 8):
            if modules[r][6] is not None:
                continue
            mt = self.grid.get_module_type(6, r)
            if mt == ModuleType.TIMING:
                val = (r - 8) % 2
                color = 0 if val == 0 else 1
                modules[r][6] = RenderedModule(
                    col=6, row=r,
                    category=ModuleCategory.AUXILIARY,
                    color_a=color, color_b=color,
                    shape_index=None,
                )

    def _fill_format_info(self, modules: List[List[RenderedModule]], fmt_bits: List[int]) -> None:
        """填充 Format Info（两份冗余）。"""
        positions = self.grid.format_info_positions  # [copy1, copy2]
        for copy_idx, coords in enumerate(positions):
            for bit_idx, (c, r) in enumerate(coords):
                if bit_idx >= 14:
                    break
                val = fmt_bits[bit_idx]
                color = 0 if val == 1 else 1  # 1=黑(0), 0=白(1)
                if modules[r][c] is None:
                    modules[r][c] = RenderedModule(
                        col=c, row=r,
                        category=ModuleCategory.AUXILIARY,
                        color_a=color, color_b=color,
                        shape_index=None,
                    )
