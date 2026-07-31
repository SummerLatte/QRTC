"""
符号子系统：符号编码/解码。

符号 = 颜色对编号 × 图形数 + 图形编号
符号值域 [0, M-1]，M = C(颜色数, 2) × 图形数

符号层向数据层交付符号值和 M，不交付颜色对、图形等视觉细节。
"""

from __future__ import annotations

from dataclasses import dataclass

from .colors import ColorPalette, ColorPair
from .shapes import Shape, ShapeSet


@dataclass(frozen=True)
class SymbolComponents:
    """符号分解后的视觉组件：颜色对 + 图形。"""
    color_pair: ColorPair
    shape: Shape
    color_pair_index: int
    shape_index: int

    @property
    def color_a(self) -> int:
        """区域 A 填入的颜色编号（编号小的颜色）。"""
        return self.color_pair.i

    @property
    def color_b(self) -> int:
        """区域 B 填入的颜色编号（编号大的颜色）。"""
        return self.color_pair.j


class SymbolSpace:
    """
    符号空间：由颜色调色板和图形集合共同定义。

    符号总数 M = C(颜色数, 2) × 图形数
    提供符号值 ↔ 视觉组件的双向转换。
    """

    def __init__(self, palette: ColorPalette, shape_set: ShapeSet) -> None:
        self.palette = palette
        self.shape_set = shape_set
        self.n_pairs = palette.n_pairs
        self.n_shapes = shape_set.n_shapes

    @property
    def M(self) -> int:
        """符号总数 = C(颜色数, 2) × 图形数。"""
        return self.n_pairs * self.n_shapes

    def encode(self, color_pair_index: int, shape_index: int) -> int:
        """符号值 = 颜色对编号 × 图形数 + 图形编号。"""
        if not 0 <= color_pair_index < self.n_pairs:
            raise ValueError(f"颜色对编号越界: {color_pair_index}")
        if not 0 <= shape_index < self.n_shapes:
            raise ValueError(f"图形编号越界: {shape_index}")
        return color_pair_index * self.n_shapes + shape_index

    def decode(self, symbol_val: int) -> SymbolComponents:
        """从符号值还原颜色对编号和图形编号。"""
        if not 0 <= symbol_val < self.M:
            raise ValueError(f"符号值越界: {symbol_val}, 范围 [0, {self.M})")
        color_pair_index = symbol_val // self.n_shapes
        shape_index = symbol_val % self.n_shapes
        return SymbolComponents(
            color_pair=self.palette.get_pair(color_pair_index),
            shape=self.shape_set.get_shape(shape_index),
            color_pair_index=color_pair_index,
            shape_index=shape_index,
        )

    def encode_components(self, comp: SymbolComponents) -> int:
        """从视觉组件编码为符号值。"""
        return self.encode(comp.color_pair_index, comp.shape_index)

    def get_colors_for_symbol(self, symbol_val: int) -> tuple[int, int, "Shape"]:
        """返回 (color_a, color_b, shape) 供渲染使用。"""
        comp = self.decode(symbol_val)
        return comp.color_a, comp.color_b, comp.shape
