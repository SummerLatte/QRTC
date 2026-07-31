"""
颜色子系统：颜色定义、颜色档位、颜色对。

颜色数可变，3 档（2/4/8 色），采用 JAB Code（ISO/IEC 23634）配色。
颜色对 (i, j) 由两个不同颜色编号组成，i < j，按字典序排列。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple


# RGB 立方体顶点，编号 0-7
_COLOR_TABLE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),        # 黑 Black
    1: (255, 255, 255),  # 白 White
    2: (255, 0, 0),      # 红 Red
    3: (0, 255, 0),      # 绿 Green
    4: (0, 0, 255),      # 蓝 Blue
    5: (255, 255, 0),    # 黄 Yellow
    6: (0, 255, 255),    # 青 Cyan
    7: (255, 0, 255),    # 品红 Magenta
}

# 颜色档位 → 该档位下可用颜色编号列表（低档为高档子集）
COLOR_LEVELS: Dict[int, List[int]] = {
    2: [0, 1],
    4: [0, 1, 2, 3],
    8: [0, 1, 2, 3, 4, 5, 6, 7],
}

# 颜色档位编码（Format Info bit 0-1）
COLOR_LEVEL_CODE: Dict[int, int] = {2: 0b00, 4: 0b01, 8: 0b10}
COLOR_LEVEL_DECODE: Dict[int, int] = {v: k for k, v in COLOR_LEVEL_CODE.items()}


@dataclass(frozen=True)
class ColorPair:
    """颜色对：两个不同颜色编号，i < j。"""
    i: int  # 编号小的颜色
    j: int  # 编号大的颜色

    def __post_init__(self) -> None:
        if self.i >= self.j:
            raise ValueError(f"颜色对要求 i < j, 得到 ({self.i}, {self.j})")

    def __iter__(self):
        yield self.i
        yield self.j

    def __repr__(self) -> str:
        return f"ColorPair({self.i}, {self.j})"


class ColorPalette:
    """
    颜色调色板：给定颜色档位，提供颜色列表、颜色对列表、RGB 查询。

    可通过子类化或传入自定义颜色表来扩展新的颜色档位。
    """

    def __init__(self, n_colors: int, color_table: Dict[int, Tuple[int, int, int]] | None = None) -> None:
        if n_colors not in COLOR_LEVELS:
            raise ValueError(f"不支持的颜色档位: {n_colors}, 可选: {list(COLOR_LEVELS.keys())}")
        self.n_colors = n_colors
        self.color_ids: List[int] = list(COLOR_LEVELS[n_colors])
        self._table = color_table if color_table is not None else _COLOR_TABLE
        self._pairs: List[ColorPair] = [
            ColorPair(i, j) for i, j in combinations(self.color_ids, 2)
        ]

    @property
    def n_pairs(self) -> int:
        """颜色对总数 = C(n, 2)。"""
        return len(self._pairs)

    @property
    def pairs(self) -> List[ColorPair]:
        """颜色对列表，按字典序排列。"""
        return list(self._pairs)

    def get_pair(self, index: int) -> ColorPair:
        """按编号获取颜色对。"""
        if not 0 <= index < self.n_pairs:
            raise IndexError(f"颜色对编号越界: {index}, 范围 [0, {self.n_pairs})")
        return self._pairs[index]

    def get_pair_index(self, pair: ColorPair) -> int:
        """从颜色对获取编号（线性查找，颜色对数量少）。"""
        for idx, p in enumerate(self._pairs):
            if p.i == pair.i and p.j == pair.j:
                return idx
        raise ValueError(f"颜色对不存在: {pair}")

    def get_rgb(self, color_id: int) -> Tuple[int, int, int]:
        """获取颜色编号对应的 RGB 值。"""
        if color_id not in self._table:
            raise KeyError(f"颜色编号不存在: {color_id}")
        return self._table[color_id]

    @property
    def code(self) -> int:
        """Format Info 中的颜色档位编码（2 bits）。"""
        return COLOR_LEVEL_CODE[self.n_colors]

    @staticmethod
    def from_code(code: int) -> "ColorPalette":
        """从 Format Info 编码构建调色板。"""
        if code not in COLOR_LEVEL_DECODE:
            raise ValueError(f"无效颜色档位编码: {code:#04b}")
        return ColorPalette(COLOR_LEVEL_DECODE[code])
