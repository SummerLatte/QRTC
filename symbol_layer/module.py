"""
模块类型定义。

模块（Module）是 Cimbar 的最小单元，为正方形区域，和像素无关。
分为两类：
- 辅助模块：纯黑或纯白，不承载数据，用于结构定位
- 数据模块：由两种对等颜色 + 一种图形组成，承载一个符号
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class ModuleCategory(IntEnum):
    """模块大类。"""
    AUXILIARY = 0  # 辅助模块（纯黑/纯白）
    DATA = 1       # 数据模块（颜色对 + 图形）


@dataclass(frozen=True)
class RenderedModule:
    """
    渲染后的模块描述：记录该模块应如何绘制。

    辅助模块：color_a = color_b = 黑或白, shape = None
    数据模块：color_a/color_b 为两种颜色编号, shape_index 为图形编号
    """
    col: int
    row: int
    category: ModuleCategory
    color_a: int   # 辅助模块时 = color_b = 0(黑) 或 1(白)
    color_b: int
    shape_index: Optional[int]  # 辅助模块为 None

    @property
    def is_auxiliary(self) -> bool:
        return self.category == ModuleCategory.AUXILIARY

    @property
    def is_data(self) -> bool:
        return self.category == ModuleCategory.DATA
