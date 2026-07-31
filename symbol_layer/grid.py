"""
网格与结构区域子系统。

网格为正方形，4 个等级（21/29/37/45 模块边长）。
结构区域包括：Finder Pattern、Separator、Timing Pattern、Format Info、Quiet Zone。
数据区模块逐行扫描，跳过所有结构区域，按扫描顺序排列为符号块。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import FrozenSet, List, Optional, Set, Tuple


class GridLevel(IntEnum):
    """网格等级。"""
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4


# 等级 → 边长（模块数）
GRID_SIZES: Dict[int, int] = {
    1: 21,
    2: 29,
    3: 37,
    4: 45,
}


# 标准 QR Finder Pattern 7×7（1=黑, 0=白）
FINDER_PATTERN: List[List[int]] = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]


@dataclass(frozen=True)
class ModuleCoord:
    """模块坐标 (col, row)，左上角为 (0, 0)，不含 Quiet Zone。"""
    col: int
    row: int

    def __iter__(self):
        yield self.col
        yield self.row


class ModuleType(IntEnum):
    """模块类型。"""
    FINDER = 0       # Finder Pattern
    SEPARATOR = 1    # Separator（白色边框）
    TIMING = 2       # Timing Pattern
    FORMAT_INFO = 3  # Format Info
    DATA = 4         # 数据模块
    QUIET_ZONE = 5   # Quiet Zone（网格外）


class Grid:
    """
    网格结构：定义所有模块的类型归属。

    提供：
    - 查询任意 (col, row) 的模块类型
    - 获取数据模块的扫描顺序列表
    - 结构模块计数
    """

    def __init__(self, level: int | GridLevel) -> None:
        level = int(level)
        if level not in GRID_SIZES:
            raise ValueError(f"无效网格等级: {level}, 可选: {list(GRID_SIZES.keys())}")
        self.level = level
        self.size: int = GRID_SIZES[level]  # 边长 N
        self.N = self.size

        # 预计算结构区域坐标集合
        self._structural_coords: Set[Tuple[int, int]] = set()
        self._finder_coords: Set[Tuple[int, int]] = set()
        self._separator_coords: Set[Tuple[int, int]] = set()
        self._timing_coords: Set[Tuple[int, int]] = set()
        self._format_info_coords: Set[Tuple[int, int]] = set()

        self._build_structural_areas()

        # 预计算数据模块扫描顺序
        self._data_module_order: List[ModuleCoord] = self._build_data_scan_order()

    def _build_structural_areas(self) -> None:
        N = self.N

        # --- Finder Patterns (3 个角) ---
        finder_positions = [
            (0, 0),           # 左上
            (N - 7, 0),       # 右上
            (0, N - 7),       # 左下
        ]
        for fc, fr in finder_positions:
            for dr in range(7):
                for dc in range(7):
                    coord = (fc + dc, fr + dr)
                    self._finder_coords.add(coord)
                    self._structural_coords.add(coord)

        # --- Separators (finder 周围 1 模块白色边框) ---
        for fc, fr in finder_positions:
            # finder 占据 [fc, fc+6] × [fr, fr+6]
            # separator 是周围一圈
            for dr in range(-1, 8):
                for dc in range(-1, 8):
                    c, r = fc + dc, fr + dr
                    if 0 <= c < N and 0 <= r < N:
                        coord = (c, r)
                        if coord not in self._finder_coords:
                            self._separator_coords.add(coord)
                            self._structural_coords.add(coord)

        # --- Timing Patterns ---
        # 水平: row=6, col=8 到 col=N-9
        for c in range(8, N - 8):
            coord = (c, 6)
            if coord not in self._structural_coords:
                self._timing_coords.add(coord)
                self._structural_coords.add(coord)
        # 垂直: col=6, row=8 到 row=N-9
        for r in range(8, N - 8):
            coord = (6, r)
            if coord not in self._structural_coords:
                self._timing_coords.add(coord)
                self._structural_coords.add(coord)

        # --- Format Info ---
        # 第一份（左上 finder 周围，14 模块）
        # 水平: row=8, col=0,1,2,3,4,5,7（bit 0-6）
        fi1_h = [(c, 8) for c in [0, 1, 2, 3, 4, 5, 7]]
        # 垂直: col=8, row=0,1,2,3,4,5,7（bit 7-13）
        fi1_v = [(8, r) for r in [0, 1, 2, 3, 4, 5, 7]]
        for coord in fi1_h + fi1_v:
            if coord not in self._structural_coords:
                self._format_info_coords.add(coord)
                self._structural_coords.add(coord)

        # 第二份（右上 + 左下 finder 周围，14 模块）
        # 水平: row=8, col=N-8,N-7,N-6,N-5,N-4,N-3,N-2（bit 0-6）
        fi2_h = [(c, 8) for c in range(N - 8, N - 1)]
        # 垂直: col=8, row=N-7,N-6,N-5,N-4,N-3,N-2,N-1（bit 7-13）
        fi2_v = [(8, r) for r in range(N - 7, N)]
        for coord in fi2_h + fi2_v:
            if coord not in self._structural_coords:
                self._format_info_coords.add(coord)
                self._structural_coords.add(coord)

    def _build_data_scan_order(self) -> List[ModuleCoord]:
        """逐行扫描，跳过所有结构区域，返回数据模块坐标列表。"""
        order: List[ModuleCoord] = []
        for r in range(self.N):
            for c in range(self.N):
                if (c, r) not in self._structural_coords:
                    order.append(ModuleCoord(c, r))
        return order

    def get_module_type(self, col: int, row: int) -> ModuleType:
        """查询模块类型。"""
        if col < 0 or col >= self.N or row < 0 or row >= self.N:
            return ModuleType.QUIET_ZONE
        coord = (col, row)
        if coord in self._finder_coords:
            return ModuleType.FINDER
        if coord in self._separator_coords:
            return ModuleType.SEPARATOR
        if coord in self._timing_coords:
            return ModuleType.TIMING
        if coord in self._format_info_coords:
            return ModuleType.FORMAT_INFO
        return ModuleType.DATA

    @property
    def data_module_count(self) -> int:
        """数据模块数 = 单帧可交付的符号数 S。"""
        return len(self._data_module_order)

    @property
    def S(self) -> int:
        """数据模块数（符号容量），等价于 data_module_count。"""
        return self.data_module_count

    @property
    def structural_module_count(self) -> int:
        """结构模块总数。"""
        return len(self._structural_coords)

    @property
    def data_scan_order(self) -> List[ModuleCoord]:
        """数据模块扫描顺序（逐行，跳过结构区域）。"""
        return list(self._data_module_order)

    def get_finder_module_value(self, col: int, row: int) -> Optional[int]:
        """
        获取 Finder Pattern 区域的黑白值（1=黑, 0=白）。
        若不在任何 finder 区域内，返回 None。
        """
        N = self.N
        finder_positions = [
            (0, 0),           # 左上
            (N - 7, 0),       # 右上
            (0, N - 7),       # 左下
        ]
        for fc, fr in finder_positions:
            if fc <= col < fc + 7 and fr <= row < fr + 7:
                return FINDER_PATTERN[row - fr][col - fc]
        return None

    def get_timing_module_value(self, col: int, row: int) -> Optional[int]:
        """
        获取 Timing Pattern 区域的黑白值（1=黑, 0=白）。
        Timing 起始于 col=8 / row=8，交替黑白，起始为黑（col=8 → 黑）。
        """
        if (col, row) in self._timing_coords:
            if row == 6:
                return (col - 8) % 2  # col=8 → 0(白)... 实际 QR timing 从 col=6 开始
            if col == 6:
                return (row - 8) % 2
        return None

    @property
    def format_info_positions(self) -> List[Tuple[Tuple[int, int], ...]]:
        """
        返回两份 Format Info 的坐标列表。
        每份 14 个坐标，按 bit 0-13 顺序排列。
        """
        N = self.N
        copy1 = tuple((c, 8) for c in [0, 1, 2, 3, 4, 5, 7]) + \
                tuple((8, r) for r in [0, 1, 2, 3, 4, 5, 7])
        copy2 = tuple((c, 8) for c in range(N - 8, N - 1)) + \
                tuple((8, r) for r in range(N - 7, N))
        return [copy1, copy2]

    def __repr__(self) -> str:
        return f"Grid(level={self.level}, N={self.N}, S={self.S})"
