"""
图形子系统：图形定义、图形档位。

图形定义两种颜色（A/B）的空间分布方式。
当前已定义 4 图形档（4 个对角三角形），其余档位待定义。

扩展方式：继承 ShapeSet 或直接注册新的图形档位。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple


class ShapeLevel:
    """图形档位常量。"""
    S2 = 2
    S4 = 4
    S8 = 8
    S16 = 16


# 图形档位编码（Format Info bit 2-3）
SHAPE_LEVEL_CODE: Dict[int, int] = {2: 0b00, 4: 0b01, 8: 0b10, 16: 0b11}
SHAPE_LEVEL_DECODE: Dict[int, int] = {v: k for k, v in SHAPE_LEVEL_CODE.items()}


@dataclass(frozen=True)
class Shape:
    """
    图形定义：名称、区域 A 判定函数、区域 B 判定函数。

    判定函数接收 (col, row) 模块内归一化坐标 [0, 1)，返回 True 表示该位置属于对应区域。
    """
    index: int
    name: str
    region_a: Callable[[float, float], bool]
    region_b: Callable[[float, float], bool]

    def __repr__(self) -> str:
        return f"Shape({self.index}, '{self.name}')"


# ---- 4 图形档：4 个对角三角形 ----

def _top_left(x: float, y: float) -> bool:
    return x + y < 1.0

def _bottom_right(x: float, y: float) -> bool:
    return x + y >= 1.0

def _top_right(x: float, y: float) -> bool:
    return (1.0 - x) + y < 1.0

def _bottom_left(x: float, y: float) -> bool:
    return (1.0 - x) + y >= 1.0


_SHAPE_SETS: Dict[int, List[Shape]] = {
    ShapeLevel.S2: [
        Shape(0, "◣ 左上三角", _top_left,     _bottom_right),
        Shape(1, "◢ 右下三角", _bottom_right,  _top_left),
    ],
    ShapeLevel.S4: [
        Shape(0, "◣ 左上三角", _top_left,     _bottom_right),
        Shape(1, "◢ 右下三角", _bottom_right,  _top_left),
        Shape(2, "◤ 右上三角", _top_right,    _bottom_left),
        Shape(3, "◥ 左下三角", _bottom_left,   _top_right),
    ],
}


def register_shape_set(n_shapes: int, shapes: List[Shape]) -> None:
    """注册新的图形档位，供扩展使用。"""
    if len(shapes) != n_shapes:
        raise ValueError(f"图形数量不匹配: 期望 {n_shapes}, 实际 {len(shapes)}")
    for i, s in enumerate(shapes):
        if s.index != i:
            raise ValueError(f"图形编号不连续: 期望 index={i}, 实际 index={s.index}")
    _SHAPE_SETS[n_shapes] = list(shapes)


class ShapeSet:
    """
    图形集合：给定图形档位，提供图形列表查询。

    可通过 register_shape_set() 扩展新的图形档位。
    """

    def __init__(self, n_shapes: int) -> None:
        if n_shapes not in _SHAPE_SETS:
            raise NotImplementedError(
                f"图形档位 {n_shapes} 尚未定义, 已定义: {list(_SHAPE_SETS.keys())}"
            )
        self.n_shapes = n_shapes
        self._shapes: List[Shape] = list(_SHAPE_SETS[n_shapes])

    @property
    def shapes(self) -> List[Shape]:
        return list(self._shapes)

    def get_shape(self, index: int) -> Shape:
        if not 0 <= index < self.n_shapes:
            raise IndexError(f"图形编号越界: {index}, 范围 [0, {self.n_shapes})")
        return self._shapes[index]

    @property
    def code(self) -> int:
        """Format Info 中的图形档位编码（2 bits）。"""
        return SHAPE_LEVEL_CODE[self.n_shapes]

    @staticmethod
    def from_code(code: int) -> "ShapeSet":
        """从 Format Info 编码构建图形集合。"""
        if code not in SHAPE_LEVEL_DECODE:
            raise ValueError(f"无效图形档位编码: {code:#04b}")
        return ShapeSet(SHAPE_LEVEL_DECODE[code])
