"""
符号层单元测试。

覆盖范围：
- colors: 颜色调色板、颜色对、颜色档位编码
- shapes: 图形集合、区域判定函数、图形档位编码
- symbol: 符号空间 encode/decode 往返
- grid: 网格尺寸、结构区域、数据模块计数
- format_info: BCH(14,4) 编解码、1 位纠错
- module: 模块类型、RenderedModule
- encoder: 符号块 → 帧模块描述矩阵
- decoder: Mock sampler 往返解码
- 集成: encoder + decoder 完整往返
"""

import pytest
from itertools import combinations

from symbol_layer import (
    ColorPalette, ColorPair, COLOR_LEVELS, COLOR_LEVEL_CODE,
    ShapeSet, Shape, ShapeLevel, SHAPE_LEVEL_CODE,
    SymbolSpace, SymbolComponents,
    Grid, GridLevel, GRID_SIZES, ModuleType, ModuleCoord, FINDER_PATTERN,
    FormatInfo,
    ModuleCategory, RenderedModule,
    SymbolEncoder, RenderedFrame,
    SymbolDecoder, DecodedFrame, DecodedSymbol, ModuleSampler,
)
from symbol_layer.colors import _COLOR_TABLE
from symbol_layer.shapes import _top_left, _bottom_right, _top_right, _bottom_left


# ======================================================================
# 1. Colors 测试
# ======================================================================

class TestColorPair:
    def test_valid_pair(self):
        p = ColorPair(0, 1)
        assert p.i == 0 and p.j == 1

    def test_equal_colors_rejected(self):
        with pytest.raises(ValueError):
            ColorPair(2, 2)

    def test_reversed_rejected(self):
        with pytest.raises(ValueError):
            ColorPair(3, 1)

    def test_iterable(self):
        p = ColorPair(1, 3)
        assert list(p) == [1, 3]

    def test_frozen(self):
        p = ColorPair(0, 1)
        with pytest.raises(Exception):
            p.i = 5  # frozen dataclass


class TestColorPalette:
    @pytest.mark.parametrize("n_colors", [2, 4, 8])
    def test_n_colors(self, n_colors):
        pal = ColorPalette(n_colors)
        assert pal.n_colors == n_colors
        assert pal.color_ids == COLOR_LEVELS[n_colors]

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            ColorPalette(3)
        with pytest.raises(ValueError):
            ColorPalette(16)

    @pytest.mark.parametrize("n_colors,expected", [
        (2, 1),   # C(2,2) = 1
        (4, 6),   # C(4,2) = 6
        (8, 28),  # C(8,2) = 28
    ])
    def test_n_pairs(self, n_colors, expected):
        pal = ColorPalette(n_colors)
        assert pal.n_pairs == expected

    def test_pairs_ordering(self):
        pal = ColorPalette(4)
        expected = [ColorPair(i, j) for i, j in combinations(range(4), 2)]
        assert pal.pairs == expected

    def test_get_pair_index_roundtrip(self):
        pal = ColorPalette(8)
        for idx in range(pal.n_pairs):
            pair = pal.get_pair(idx)
            assert pal.get_pair_index(pair) == idx

    def test_get_pair_out_of_range(self):
        pal = ColorPalette(2)
        with pytest.raises(IndexError):
            pal.get_pair(1)

    def test_get_pair_index_not_found(self):
        pal = ColorPalette(2)  # 只有 (0,1)
        with pytest.raises(ValueError):
            pal.get_pair_index(ColorPair(2, 3))

    @pytest.mark.parametrize("color_id,expected_rgb", [
        (0, (0, 0, 0)),
        (1, (255, 255, 255)),
        (2, (255, 0, 0)),
        (3, (0, 255, 0)),
        (4, (0, 0, 255)),
        (5, (255, 255, 0)),
        (6, (0, 255, 255)),
        (7, (255, 0, 255)),
    ])
    def test_get_rgb(self, color_id, expected_rgb):
        pal = ColorPalette(8)
        assert pal.get_rgb(color_id) == expected_rgb

    def test_subset_property(self):
        pal2 = ColorPalette(2)
        pal4 = ColorPalette(4)
        pal8 = ColorPalette(8)
        assert set(pal2.color_ids).issubset(set(pal4.color_ids))
        assert set(pal4.color_ids).issubset(set(pal8.color_ids))

    @pytest.mark.parametrize("n_colors,expected_code", [
        (2, 0b00), (4, 0b01), (8, 0b10),
    ])
    def test_code(self, n_colors, expected_code):
        pal = ColorPalette(n_colors)
        assert pal.code == expected_code

    @pytest.mark.parametrize("code,n_colors", [
        (0b00, 2), (0b01, 4), (0b10, 8),
    ])
    def test_from_code(self, code, n_colors):
        pal = ColorPalette.from_code(code)
        assert pal.n_colors == n_colors

    def test_from_code_invalid(self):
        with pytest.raises(ValueError):
            ColorPalette.from_code(0b11)


# ======================================================================
# 2. Shapes 测试
# ======================================================================

class TestShapeRegions:
    """测试图形区域判定函数。"""

    def test_top_left(self):
        assert _top_left(0.1, 0.1) is True
        assert _top_left(0.9, 0.9) is False
        assert _top_left(0.3, 0.3) is True   # 0.3+0.3=0.6 < 1.0

    def test_bottom_right(self):
        assert _bottom_right(0.9, 0.9) is True
        assert _bottom_right(0.1, 0.1) is False
        assert _bottom_right(0.6, 0.6) is True  # 0.6+0.6=1.2 >= 1.0

    def test_top_right(self):
        assert _top_right(0.9, 0.1) is True   # (1-0.9)+0.1 = 0.2 < 1.0
        assert _top_right(0.1, 0.9) is False  # (1-0.1)+0.9 = 1.8 >= 1.0

    def test_bottom_left(self):
        assert _bottom_left(0.1, 0.9) is True   # (1-0.1)+0.9 = 1.8 >= 1.0
        assert _bottom_left(0.9, 0.1) is False  # (1-0.9)+0.1 = 0.2 < 1.0

    def test_complementarity(self):
        """区域 A 和区域 B 应互补（覆盖整个模块）。"""
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for y in [0.1, 0.3, 0.5, 0.7, 0.9]:
                assert _top_left(x, y) != _bottom_right(x, y)
                assert _top_right(x, y) != _bottom_left(x, y)


class TestShapeSet:
    def test_s2(self):
        ss = ShapeSet(2)
        assert ss.n_shapes == 2
        shapes = ss.shapes
        assert len(shapes) == 2
        assert shapes[0].index == 0
        assert shapes[1].index == 1

    def test_s4(self):
        ss = ShapeSet(4)
        assert ss.n_shapes == 4
        for i, s in enumerate(ss.shapes):
            assert s.index == i

    def test_s8_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ShapeSet(8)

    def test_s16_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ShapeSet(16)

    def test_get_shape_out_of_range(self):
        ss = ShapeSet(4)
        with pytest.raises(IndexError):
            ss.get_shape(4)
        with pytest.raises(IndexError):
            ss.get_shape(-1)

    @pytest.mark.parametrize("n_shapes,expected_code", [
        (2, 0b00), (4, 0b01),
    ])
    def test_code(self, n_shapes, expected_code):
        ss = ShapeSet(n_shapes)
        assert ss.code == expected_code

    @pytest.mark.parametrize("code,n_shapes", [
        (0b00, 2), (0b01, 4),
    ])
    def test_from_code(self, code, n_shapes):
        ss = ShapeSet.from_code(code)
        assert ss.n_shapes == n_shapes

    def test_from_code_invalid(self):
        # 0b11 对应 16 图形档，尚未实现
        with pytest.raises(NotImplementedError):
            ShapeSet.from_code(0b11)


# ======================================================================
# 3. Symbol 测试
# ======================================================================

class TestSymbolSpace:
    @pytest.mark.parametrize("n_colors,n_shapes,expected_M", [
        (2, 2, 1 * 2),    # C(2,2)=1, 2 shapes
        (2, 4, 1 * 4),
        (4, 2, 6 * 2),    # C(4,2)=6
        (4, 4, 6 * 4),
        (8, 2, 28 * 2),   # C(8,2)=28
        (8, 4, 28 * 4),
    ])
    def test_M(self, n_colors, n_shapes, expected_M):
        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        space = SymbolSpace(pal, ss)
        assert space.M == expected_M

    @pytest.mark.parametrize("n_colors,n_shapes", [
        (2, 2), (2, 4), (4, 2), (4, 4), (8, 2), (8, 4),
    ])
    def test_encode_decode_roundtrip(self, n_colors, n_shapes):
        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        space = SymbolSpace(pal, ss)
        for sv in range(space.M):
            comp = space.decode(sv)
            assert comp.color_pair_index * n_shapes + comp.shape_index == sv
            re_encoded = space.encode(comp.color_pair_index, comp.shape_index)
            assert re_encoded == sv

    def test_encode_out_of_range(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        space = SymbolSpace(pal, ss)
        with pytest.raises(ValueError):
            space.encode(6, 0)  # pair_index 越界
        with pytest.raises(ValueError):
            space.encode(0, 4)  # shape_index 越界

    def test_decode_out_of_range(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        space = SymbolSpace(pal, ss)
        with pytest.raises(ValueError):
            space.decode(-1)
        with pytest.raises(ValueError):
            space.decode(space.M)

    def test_color_a_is_smaller(self):
        """区域 A 填入编号小的颜色。"""
        pal = ColorPalette(8)
        ss = ShapeSet(4)
        space = SymbolSpace(pal, ss)
        for sv in range(space.M):
            comp = space.decode(sv)
            assert comp.color_a < comp.color_b

    def test_encode_components(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        space = SymbolSpace(pal, ss)
        sv = 7
        comp = space.decode(sv)
        assert space.encode_components(comp) == sv

    def test_get_colors_for_symbol(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        space = SymbolSpace(pal, ss)
        sv = 5  # pair=1(黑,红), shape=1(◢)
        color_a, color_b, shape = space.get_colors_for_symbol(sv)
        assert color_a == 0  # 黑
        assert color_b == 2  # 红
        assert shape.index == 1

    def test_symbol_table_appendix_b(self):
        """验证文档附录 B 的 4 色+4 图形符号编码表。"""
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        space = SymbolSpace(pal, ss)
        # 验证几个关键点
        # 符号 0: pair=0(黑,白), shape=0(◣)
        comp0 = space.decode(0)
        assert comp0.color_pair_index == 0
        assert comp0.shape_index == 0
        assert comp0.color_a == 0 and comp0.color_b == 1
        # 符号 12: pair=3(白,红), shape=0(◣)
        comp12 = space.decode(12)
        assert comp12.color_pair_index == 3
        assert comp12.shape_index == 0
        assert comp12.color_a == 1 and comp12.color_b == 2
        # 符号 23: pair=5(红,绿), shape=3(◥)
        comp23 = space.decode(23)
        assert comp23.color_pair_index == 5
        assert comp23.shape_index == 3
        assert comp23.color_a == 2 and comp23.color_b == 3


# ======================================================================
# 4. Grid 测试
# ======================================================================

class TestGrid:
    @pytest.mark.parametrize("level,expected_size", [
        (1, 21), (2, 29), (3, 37), (4, 45), (5, 53), (6, 61), (7, 69), (8, 77), (9, 85),
        (10, 93), (11, 101), (12, 109), (13, 117), (14, 125), (15, 133),
    ])
    def test_grid_sizes(self, level, expected_size):
        grid = Grid(level)
        assert grid.N == expected_size
        assert grid.size == expected_size

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            Grid(0)
        with pytest.raises(ValueError):
            Grid(16)

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    def test_level_plus_8(self, level):
        if level < 15:
            g1 = Grid(level)
            g2 = Grid(level + 1)
            assert g2.N - g1.N == 8

    @pytest.mark.parametrize("level,expected_data", [
        (1, 216), (2, 616), (3, 1144), (4, 1800), (5, 2584), (6, 3496), (7, 4536), (8, 5704), (9, 7000),
        (10, 8424), (11, 9976), (12, 11656), (13, 13464), (14, 15400), (15, 17464),
    ])
    def test_data_module_count(self, level, expected_data):
        """文档 8.1: 数据模块数 = N² - 225。"""
        grid = Grid(level)
        assert grid.data_module_count == expected_data
        assert grid.S == expected_data

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    def test_structural_module_count(self, level):
        """结构模块总数应为 225。"""
        grid = Grid(level)
        assert grid.structural_module_count == 225

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    def test_total_equals_structural_plus_data(self, level):
        grid = Grid(level)
        assert grid.N ** 2 == grid.structural_module_count + grid.data_module_count

    def test_finder_pattern_shape(self):
        assert len(FINDER_PATTERN) == 7
        for row in FINDER_PATTERN:
            assert len(row) == 7

    def test_finder_pattern_content(self):
        """验证标准 QR Finder Pattern 内容。"""
        expected = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
        assert FINDER_PATTERN == expected

    def test_module_type_finder_tl(self):
        grid = Grid(1)
        assert grid.get_module_type(0, 0) == ModuleType.FINDER
        assert grid.get_module_type(6, 6) == ModuleType.FINDER

    def test_module_type_finder_tr(self):
        grid = Grid(1)
        N = grid.N
        assert grid.get_module_type(N - 1, 0) == ModuleType.FINDER
        assert grid.get_module_type(N - 7, 0) == ModuleType.FINDER

    def test_module_type_finder_bl(self):
        grid = Grid(1)
        N = grid.N
        assert grid.get_module_type(0, N - 1) == ModuleType.FINDER

    def test_module_type_separator(self):
        grid = Grid(1)
        # TL finder 右侧的 separator
        assert grid.get_module_type(7, 0) == ModuleType.SEPARATOR
        # TL finder 下方的 separator
        assert grid.get_module_type(0, 7) == ModuleType.SEPARATOR

    def test_module_type_l_corner(self):
        grid = Grid(1)
        N = grid.N
        assert grid.get_module_type(N - 1, N - 1) == ModuleType.L_CORNER
        assert grid.get_module_type(N - 1, N - 2) == ModuleType.L_CORNER
        assert grid.get_module_type(N - 2, N - 1) == ModuleType.L_CORNER

    def test_module_type_data(self):
        grid = Grid(1)
        # 一个明确在数据区的位置
        assert grid.get_module_type(10, 10) == ModuleType.DATA

    def test_module_type_quiet_zone(self):
        grid = Grid(1)
        assert grid.get_module_type(-1, 0) == ModuleType.QUIET_ZONE
        assert grid.get_module_type(0, -1) == ModuleType.QUIET_ZONE
        assert grid.get_module_type(grid.N, 0) == ModuleType.QUIET_ZONE

    def test_data_scan_order_length(self):
        grid = Grid(2)
        assert len(grid.data_scan_order) == grid.data_module_count

    def test_data_scan_order_row_major(self):
        """扫描顺序应为逐行扫描。"""
        grid = Grid(1)
        order = grid.data_scan_order
        for i in range(len(order) - 1):
            # 同一行内 col 递增，或换行
            if order[i].row == order[i + 1].row:
                assert order[i].col < order[i + 1].col
            else:
                assert order[i].row < order[i + 1].row

    def test_data_scan_order_no_structural(self):
        """扫描顺序中不应包含任何结构模块。"""
        grid = Grid(3)
        for coord in grid.data_scan_order:
            assert grid.get_module_type(coord.col, coord.row) == ModuleType.DATA

    def test_format_info_positions(self):
        grid = Grid(2)
        positions = grid.format_info_positions
        assert len(positions) == 2
        for pos in positions:
            assert len(pos) == 14

    def test_get_finder_module_value(self):
        grid = Grid(1)
        # TL finder 中心 (3,3) 应为黑 (1)
        assert grid.get_finder_module_value(3, 3) == 1
        # TL finder 内圈空白 (1,1) 应为白 (0)
        assert grid.get_finder_module_value(1, 1) == 0
        # L 型角标应为黑
        N = grid.N
        assert grid.get_finder_module_value(N - 1, N - 1) == 1
        # 数据区返回 None
        assert grid.get_finder_module_value(10, 10) is None


# ======================================================================
# 5. Format Info 测试
# ======================================================================

class TestFormatInfo:
    @pytest.mark.parametrize("color_code,shape_code", [
        (0b00, 0b00), (0b01, 0b00), (0b10, 0b00),
        (0b00, 0b01), (0b01, 0b01), (0b10, 0b01),
    ])
    def test_encode_decode_no_error(self, color_code, shape_code):
        fmt = FormatInfo.from_codes(color_code, shape_code)
        cw = fmt.encode()
        decoded, err = FormatInfo.decode(cw)
        assert err == 0
        assert decoded.color_level_code == color_code
        assert decoded.shape_level_code == shape_code

    def test_data_bits_composition(self):
        fmt = FormatInfo.from_codes(0b10, 0b01)
        # data_bits = (color << 2) | shape = (0b10 << 2) | 0b01 = 0b1001 = 9
        assert fmt.data_bits == 0b1001

    def test_to_bit_list_length(self):
        fmt = FormatInfo.from_codes(0b00, 0b00)
        bits = fmt.to_bit_list()
        assert len(bits) == 14

    def test_from_bit_list_roundtrip(self):
        fmt = FormatInfo.from_codes(0b01, 0b01)
        bits = fmt.to_bit_list()
        decoded, err = FormatInfo.from_bit_list(bits)
        assert err == 0
        assert decoded.color_level_code == 0b01
        assert decoded.shape_level_code == 0b01

    def test_from_bit_list_wrong_length(self):
        with pytest.raises(ValueError):
            FormatInfo.from_bit_list([0, 1, 0])

    @pytest.mark.parametrize("bit_pos", range(14))
    def test_single_bit_error_correction(self, bit_pos):
        """BCH(14,4) 应能纠正任意 1 位错误。"""
        fmt = FormatInfo.from_codes(0b10, 0b01)
        cw = fmt.encode()
        # 翻转第 bit_pos 位
        corrupted = cw ^ (1 << bit_pos)
        decoded, err = FormatInfo.decode(corrupted)
        assert err == 1  # 检测到并纠正了 1 位错误
        assert decoded.color_level_code == 0b10
        assert decoded.shape_level_code == 0b01

    def test_multi_bit_error_uncorrectable(self):
        """2 位以上错误应返回 -1。"""
        fmt = FormatInfo.from_codes(0b10, 0b01)
        cw = fmt.encode()
        # 翻转 2 位
        corrupted = cw ^ (1 << 3) ^ (1 << 7)
        decoded, err = FormatInfo.decode(corrupted)
        assert err == -1


# ======================================================================
# 6. Module 测试
# ======================================================================

class TestModule:
    def test_category_values(self):
        assert int(ModuleCategory.AUXILIARY) == 0
        assert int(ModuleCategory.DATA) == 1

    def test_rendered_module_auxiliary(self):
        mod = RenderedModule(
            col=0, row=0, category=ModuleCategory.AUXILIARY,
            color_a=0, color_b=0, shape_index=None,
        )
        assert mod.is_auxiliary is True
        assert mod.is_data is False

    def test_rendered_module_data(self):
        mod = RenderedModule(
            col=5, row=5, category=ModuleCategory.DATA,
            color_a=0, color_b=2, shape_index=1,
        )
        assert mod.is_auxiliary is False
        assert mod.is_data is True
        assert mod.shape_index == 1

    def test_rendered_module_frozen(self):
        mod = RenderedModule(
            col=0, row=0, category=ModuleCategory.AUXILIARY,
            color_a=0, color_b=0, shape_index=None,
        )
        with pytest.raises(Exception):
            mod.col = 1


# ======================================================================
# 7. Encoder 测试
# ======================================================================

class TestSymbolEncoder:
    @pytest.fixture
    def encoder(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        return SymbolEncoder(pal, ss, grid)

    def test_S_and_M(self, encoder):
        assert encoder.S == 216  # Grid L1 数据模块数
        assert encoder.M == 24   # C(4,2)*4 = 6*4

    def test_encode_valid(self, encoder):
        block = [i % encoder.M for i in range(encoder.S)]
        frame = encoder.encode(block)
        assert isinstance(frame, RenderedFrame)
        assert frame.N == 21

    def test_encode_wrong_length(self, encoder):
        with pytest.raises(ValueError):
            encoder.encode([0] * (encoder.S - 1))
        with pytest.raises(ValueError):
            encoder.encode([0] * (encoder.S + 1))

    def test_encode_symbol_out_of_range(self, encoder):
        block = [0] * encoder.S
        block[10] = encoder.M  # 越界
        with pytest.raises(ValueError):
            encoder.encode(block)

    def test_encode_all_modules_filled(self, encoder):
        """编码后所有模块位置不应有 None。"""
        block = [0] * encoder.S
        frame = encoder.encode(block)
        for row in range(frame.N):
            for col in range(frame.N):
                assert frame.modules[row][col] is not None, \
                    f"模块 ({col},{row}) 未填充"

    def test_encode_finder_pattern_correct(self, encoder):
        """验证编码后的 Finder Pattern 与标准一致。"""
        block = [0] * encoder.S
        frame = encoder.encode(block)
        # TL finder (0,0) - (6,6)
        for r in range(7):
            for c in range(7):
                mod = frame.get_module(c, r)
                assert mod.is_auxiliary
                expected_val = FINDER_PATTERN[r][c]
                # 1=黑(color_a=0), 0=白(color_a=1)
                expected_color = 0 if expected_val == 1 else 1
                assert mod.color_a == expected_color

    def test_encode_separator_white(self, encoder):
        """Separator 应为白色。"""
        block = [0] * encoder.S
        frame = encoder.encode(block)
        # (7, 0) 是 TL finder 右侧的 separator
        mod = frame.get_module(7, 0)
        assert mod.is_auxiliary
        assert mod.color_a == 1  # 白
        assert mod.color_b == 1

    def test_encode_l_corner_black(self, encoder):
        """L 型角标应为黑色。"""
        block = [0] * encoder.S
        frame = encoder.encode(block)
        N = frame.N
        mod = frame.get_module(N - 1, N - 1)
        assert mod.is_auxiliary
        assert mod.color_a == 0  # 黑

    def test_encode_data_module_has_shape(self, encoder):
        """数据模块应有 shape_index。"""
        block = [0] * encoder.S
        frame = encoder.encode(block)
        # 找一个数据模块
        grid = encoder.grid
        for coord in grid.data_scan_order:
            mod = frame.get_module(coord.col, coord.row)
            assert mod.is_data
            assert mod.shape_index is not None
            assert mod.color_a != mod.color_b  # 不允许纯色
            break

    def test_encode_format_info_in_frame(self, encoder):
        """验证 Format Info 被正确编码到帧中。"""
        block = [0] * encoder.S
        frame = encoder.encode(block)
        fmt = FormatInfo.from_codes(encoder.palette.code, encoder.shape_set.code)
        expected_bits = fmt.to_bit_list()

        positions = encoder.grid.format_info_positions
        for coords in positions:
            for bit_idx, (c, r) in enumerate(coords):
                if bit_idx >= 14:
                    break
                mod = frame.get_module(c, r)
                assert mod.is_auxiliary
                # bit=1 → 黑(0), bit=0 → 白(1)
                expected_color = 0 if expected_bits[bit_idx] == 1 else 1
                assert mod.color_a == expected_color, \
                    f"Format Info bit {bit_idx} at ({c},{r}): " \
                    f"expected color {expected_color}, got {mod.color_a}"

    def test_encode_data_module_colors_match_symbol(self, encoder):
        """验证数据模块的颜色与符号值对应。"""
        block = [5] * encoder.S  # 符号 5: pair=1(黑,红), shape=1(◢)
        frame = encoder.encode(block)
        grid = encoder.grid
        coord = grid.data_scan_order[0]
        mod = frame.get_module(coord.col, coord.row)
        comp = encoder.symbol_space.decode(5)
        assert mod.color_a == comp.color_a
        assert mod.color_b == comp.color_b
        assert mod.shape_index == comp.shape_index


# ======================================================================
# 8. Decoder 测试（使用 Mock Sampler）
# ======================================================================

class MockSampler(ModuleSampler):
    """
    Mock 采样器：从 RenderedFrame 直接读取模块信息，
    模拟理想无损的图像采样场景。
    """

    def __init__(self, frame: RenderedFrame, grid: Grid,
                 palette: ColorPalette, shape_set: ShapeSet) -> None:
        self._frame = frame
        self._grid = grid
        self._palette = palette
        self._shape_set = shape_set

    def sample_module_colors(self, col: int, row: int):
        mod = self._frame.get_module(col, row)
        if mod is None or mod.is_auxiliary:
            return (0, 0, 0), (0, 0, 0), 0, 0.0
        rgb_a = self._palette.get_rgb(mod.color_a)
        rgb_b = self._palette.get_rgb(mod.color_b)
        return rgb_a, rgb_b, mod.shape_index, 1.0

    def sample_auxiliary_module(self, col: int, row: int):
        mod = self._frame.get_module(col, row)
        if mod is None:
            return 1, 1.0  # 白
        # color_a=0 → 黑(val=0), color_a=1 → 白(val=1)
        return mod.color_a, 1.0


class TestSymbolDecoder:
    @pytest.fixture
    def setup(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        encoder = SymbolEncoder(pal, ss, grid)
        # 生成随机符号块
        import random
        rng = random.Random(42)
        block = [rng.randint(0, encoder.M - 1) for _ in range(encoder.S)]
        frame = encoder.encode(block)
        sampler = MockSampler(frame, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        return pal, ss, grid, encoder, block, frame, decoder

    def test_decode_returns_decoded_frame(self, setup):
        _, _, _, _, _, _, decoder = setup
        result = decoder.decode()
        assert isinstance(result, DecodedFrame)

    def test_decode_symbol_block_length(self, setup):
        _, _, grid, _, _, _, decoder = setup
        result = decoder.decode()
        assert len(result.symbol_block) == grid.S

    def test_decode_format_info(self, setup):
        pal, ss, _, _, _, _, decoder = setup
        result = decoder.decode()
        assert result.format_info.color_level_code == pal.code
        assert result.format_info.shape_level_code == ss.code

    def test_decode_roundtrip(self, setup):
        """理想无损场景下，解码结果应与原始符号块完全一致。"""
        _, _, _, _, block, _, decoder = setup
        result = decoder.decode()
        assert result.symbol_block == block

    def test_decode_confidences(self, setup):
        _, _, _, _, _, _, decoder = setup
        result = decoder.decode()
        # Mock sampler 返回置信度 1.0
        for conf in result.confidences:
            assert conf == 1.0

    def test_decode_no_erasures(self, setup):
        _, _, _, _, _, _, decoder = setup
        result = decoder.decode()
        assert not any(result.erasure_flags)

    def test_decode_M(self, setup):
        _, _, _, _, _, _, decoder = setup
        result = decoder.decode()
        assert result.M == 24  # C(4,2)*4

    def test_match_color(self, setup):
        _, _, _, _, _, _, decoder = setup
        pal = ColorPalette(4)
        # 精确颜色匹配
        assert decoder._match_color((0, 0, 0), pal) == 0       # 黑
        assert decoder._match_color((255, 255, 255), pal) == 1  # 白
        assert decoder._match_color((255, 0, 0), pal) == 2      # 红
        assert decoder._match_color((0, 255, 0), pal) == 3      # 绿
        # 近似颜色匹配
        assert decoder._match_color((10, 10, 10), pal) == 0     # 近黑
        assert decoder._match_color((250, 5, 5), pal) == 2      # 近红

    @pytest.mark.parametrize("n_shapes,shape_idx,expected", [
        (2, 0, 1), (2, 1, 0),
        (4, 0, 1), (4, 1, 0), (4, 2, 3), (4, 3, 2),
    ])
    def test_flip_shape(self, n_shapes, shape_idx, expected):
        assert SymbolDecoder._flip_shape(shape_idx, n_shapes) == expected


# ======================================================================
# 9. 集成测试：Encoder + Decoder 往返
# ======================================================================

class TestIntegrationRoundtrip:
    @pytest.mark.parametrize("n_colors,n_shapes,level", [
        (2, 2, 1), (2, 4, 1),
        (4, 2, 1), (4, 4, 1),
        (4, 4, 2), (4, 4, 3), (4, 4, 4),
        (8, 4, 1), (8, 4, 2),
    ])
    def test_full_roundtrip(self, n_colors, n_shapes, level):
        """完整往返：符号块 → 编码 → Mock 采样 → 解码 → 对比。"""
        import random
        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        grid = Grid(level)
        encoder = SymbolEncoder(pal, ss, grid)
        M = encoder.M
        S = encoder.S

        rng = random.Random(123)
        original = [rng.randint(0, M - 1) for _ in range(S)]
        frame = encoder.encode(original)

        sampler = MockSampler(frame, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        assert decoded.symbol_block == original
        assert decoded.M == M
        assert decoded.S == S

    def test_all_zero_symbols(self):
        """全 0 符号块往返。"""
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        encoder = SymbolEncoder(pal, ss, grid)
        original = [0] * encoder.S
        frame = encoder.encode(original)
        sampler = MockSampler(frame, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()
        assert decoded.symbol_block == original

    def test_all_max_symbols(self):
        """全最大值符号块往返。"""
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        encoder = SymbolEncoder(pal, ss, grid)
        original = [encoder.M - 1] * encoder.S
        frame = encoder.encode(original)
        sampler = MockSampler(frame, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()
        assert decoded.symbol_block == original

    def test_sequential_symbols(self):
        """顺序符号值往返。"""
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(2)
        encoder = SymbolEncoder(pal, ss, grid)
        original = [i % encoder.M for i in range(encoder.S)]
        frame = encoder.encode(original)
        sampler = MockSampler(frame, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()
        assert decoded.symbol_block == original
