"""符号层实现验证脚本"""
import math
import random
from symbol_layer import (
    ColorPalette, ColorPair, ShapeSet, ShapeLevel,
    SymbolSpace, Grid, GridLevel, ModuleType,
    FormatInfo, SymbolEncoder, ModuleCategory,
    SymbolDecoder, ModuleSampler, RenderedFrame,
    register_shape_set, Shape,
)
from typing import Tuple

def main():
    random.seed(42)
    print("=== 导入测试 ===")
    print("[OK] 所有模块导入成功")

    # 颜色
    print("\n=== 颜色测试 ===")
    for n in [2, 4, 8]:
        p = ColorPalette(n)
        assert p.n_pairs == math.comb(n, 2)
        print(f"[OK] {n}色档: {p.n_pairs} 个颜色对, code={p.code:#04b}")
    p4 = ColorPalette(4)
    assert p4.get_pair(0) == ColorPair(0, 1)
    assert p4.get_pair(5) == ColorPair(2, 3)
    print("[OK] 4色档颜色对与附录一致")

    # 图形
    print("\n=== 图形测试 ===")
    s4 = ShapeSet(4)
    assert s4.n_shapes == 4
    print(f"[OK] 4图形档: {s4.n_shapes} 个图形, code={s4.code:#04b}")
    for s in s4.shapes:
        print(f"     {s}")

    # 符号空间
    print("\n=== 符号空间测试 ===")
    for n_c in [2, 4, 8]:
        for n_s in [2, 4]:
            pal = ColorPalette(n_c)
            shp = ShapeSet(n_s)
            ss = SymbolSpace(pal, shp)
            M = ss.M
            for sv in range(M):
                comp = ss.decode(sv)
                assert comp.color_pair_index * n_s + comp.shape_index == sv
                re_encoded = ss.encode(comp.color_pair_index, comp.shape_index)
                assert re_encoded == sv
            print(f"[OK] {n_c}色+{n_s}图形: M={M}, 往返全部通过")

    # 网格
    print("\n=== 网格测试 ===")
    expected_data = {1: 211, 2: 595, 3: 1107, 4: 1747}
    for level in [1, 2, 3, 4]:
        g = Grid(level)
        assert g.S == expected_data[level], f"等级{level}: S={g.S} != {expected_data[level]}"
        print(f"[OK] 等级{level}: N={g.N}, S={g.S}, 结构模块={g.structural_module_count}")

    # Format Info BCH
    print("\n=== Format Info BCH 测试 ===")
    for cc in [0, 1, 2]:
        for sc in [0, 1, 2, 3]:
            fmt = FormatInfo.from_codes(cc, sc)
            cw = fmt.encode()
            fmt2, err = FormatInfo.decode(cw)
            assert err == 0
            assert fmt2.color_level_code == cc
            assert fmt2.shape_level_code == sc
    print("[OK] BCH(14,4) 编解码往返全部通过")

    # 单错误纠正
    fmt = FormatInfo.from_codes(1, 1)
    cw = fmt.encode()
    for pos in range(14):
        cw_err = cw ^ (1 << pos)
        fmt2, err = FormatInfo.decode(cw_err)
        assert err == 1, f"pos={pos}: err={err}"
        assert fmt2.color_level_code == 1
        assert fmt2.shape_level_code == 1
    print("[OK] BCH(14,4) 单错误纠正全部通过")

    # 编码器
    print("\n=== 编码器测试 ===")
    pal = ColorPalette(4)
    shp = ShapeSet(4)
    g = Grid(2)
    enc = SymbolEncoder(pal, shp, g)
    print(f"  参数: N={g.N}, S={enc.S}, M={enc.M}")

    sym_block = [random.randint(0, enc.M - 1) for _ in range(enc.S)]
    frame = enc.encode(sym_block)

    # 验证所有模块都已填充
    for r in range(g.N):
        for c in range(g.N):
            assert frame.modules[r][c] is not None, f"模块 ({c},{r}) 未填充"

    # 验证数据模块数量
    data_count = sum(1 for r in range(g.N) for c in range(g.N)
                     if frame.modules[r][c].is_data)
    assert data_count == g.S
    print(f"[OK] 编码器: {data_count} 个数据模块已填充")

    # 验证符号值正确映射
    data_coords = g.data_scan_order
    for i, coord in enumerate(data_coords):
        mod = frame.modules[coord.row][coord.col]
        assert mod.is_data
        comp = enc.symbol_space.decode(sym_block[i])
        assert mod.color_a == comp.color_a
        assert mod.color_b == comp.color_b
        assert mod.shape_index == comp.shape_index
    print("[OK] 编码器: 符号值->模块映射全部正确")

    # 全等级全档位编码测试
    print("\n=== 全等级全档位编码测试 ===")
    for level in [1, 2, 3, 4]:
        g = Grid(level)
        for n_c in [2, 4, 8]:
            for n_s in [2, 4]:
                pal = ColorPalette(n_c)
                shp = ShapeSet(n_s)
                enc = SymbolEncoder(pal, shp, g)
                block = [random.randint(0, enc.M - 1) for _ in range(enc.S)]
                frame = enc.encode(block)
                # 确认无空模块
                for r in range(g.N):
                    for c in range(g.N):
                        assert frame.modules[r][c] is not None
                print(f"  [OK] 等级{level} {n_c}色+{n_s}图形: S={enc.S}, M={enc.M}")

    print("\n=== 全部单元测试通过 ===")


# ============================================================
# E2E 测试：encode → decode 往返
# ============================================================

class MockSampler(ModuleSampler):
    """
    从 RenderedFrame 读取模块信息的模拟采样器。
    模拟理想无噪声场景下的图像采样。
    """

    def __init__(self, frame: RenderedFrame, palette: ColorPalette) -> None:
        self.frame = frame
        self.palette = palette

    def sample_module_colors(self, col: int, row: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int, float]:
        mod = self.frame.modules[row][col]
        rgb_a = self.palette.get_rgb(mod.color_a)
        rgb_b = self.palette.get_rgb(mod.color_b)
        return rgb_a, rgb_b, mod.shape_index, 1.0

    def sample_auxiliary_module(self, col: int, row: int) -> Tuple[int, float]:
        mod = self.frame.modules[row][col]
        # color_a = 0 → 黑 → val=0, color_a = 1 → 白 → val=1
        return mod.color_a, 1.0


def test_e2e():
    print("\n=== E2E 测试: encode -> decode 往返 ===")

    for level in [1, 2, 3, 4]:
        g = Grid(level)
        for n_c in [2, 4, 8]:
            for n_s in [2, 4]:
                pal = ColorPalette(n_c)
                shp = ShapeSet(n_s)
                enc = SymbolEncoder(pal, shp, g)
                M = enc.M
                S = enc.S

                # 生成随机符号块
                original_block = [random.randint(0, M - 1) for _ in range(S)]

                # 编码
                frame = enc.encode(original_block)

                # 解码（用 MockSampler 模拟采样）
                sampler = MockSampler(frame, pal)
                dec = SymbolDecoder(sampler, g)
                decoded = dec.decode()

                # 验证符号块完全一致
                assert decoded.symbol_block == original_block, \
                    f"等级{level} {n_c}色+{n_s}图形: 符号块不匹配"
                assert decoded.M == M, f"M 不匹配: {decoded.M} != {M}"
                assert decoded.S == S, f"S 不匹配: {decoded.S} != {S}"

                # 验证置信度全为 1.0（理想无噪声）
                assert all(c == 1.0 for c in decoded.confidences), "置信度应为 1.0"
                assert not any(decoded.erasure_flags), "无 erasure"

                # 验证 Format Info 一致
                assert decoded.format_info.color_level_code == pal.code
                assert decoded.format_info.shape_level_code == shp.code

                print(f"  [OK] 等级{level} {n_c}色+{n_s}图形: S={S}, M={M}, 往返一致")

    print("[OK] 全等级全档位 E2E 往返测试通过")


def test_e2e_all_symbols():
    """遍历每个符号值，确保编码→解码能正确还原每个符号。"""
    print("\n=== E2E 测试: 全符号值遍历 ===")

    for n_c in [2, 4, 8]:
        for n_s in [2, 4]:
            pal = ColorPalette(n_c)
            shp = ShapeSet(n_s)
            g = Grid(1)  # 最小网格，速度快
            enc = SymbolEncoder(pal, shp, g)
            M = enc.M
            S = enc.S

            # 构造符号块：前 M 个模块依次填入 0..M-1，其余填 0
            test_block = [0] * S
            for i in range(min(M, S)):
                test_block[i] = i

            frame = enc.encode(test_block)
            sampler = MockSampler(frame, pal)
            dec = SymbolDecoder(sampler, g)
            decoded = dec.decode()

            assert decoded.symbol_block == test_block, \
                f"{n_c}色+{n_s}图形: 全符号遍历不匹配"
            print(f"  [OK] {n_c}色+{n_s}图形: M={M}, 全符号值遍历往返一致")


def test_e2e_format_info_error_correction():
    """模拟 Format Info 单模块翻转，验证 BCH 纠错。"""
    print("\n=== E2E 测试: Format Info 纠错 ===")

    pal = ColorPalette(4)
    shp = ShapeSet(4)
    g = Grid(2)
    enc = SymbolEncoder(pal, shp, g)
    original_block = [random.randint(0, enc.M - 1) for _ in range(enc.S)]
    frame = enc.encode(original_block)

    # 正常解码
    sampler = MockSampler(frame, pal)
    dec = SymbolDecoder(sampler, g)
    decoded = dec.decode()
    assert decoded.symbol_block == original_block

    # 翻转第一份 Format Info 的 1 个模块（模拟单错误）
    fi_positions = g.format_info_positions
    c, r = fi_positions[0][0]  # 第一份第一个位置
    mod = frame.modules[r][c]
    # 翻转黑白
    flipped_color = 1 - mod.color_a
    frame.modules[r][c] = type(mod)(
        col=mod.col, row=mod.row,
        category=mod.category,
        color_a=flipped_color, color_b=flipped_color,
        shape_index=mod.shape_index,
    )

    # 解码应仍能正确还原（BCH 纠正 1 位错误，第二份冗余也可用）
    sampler2 = MockSampler(frame, pal)
    dec2 = SymbolDecoder(sampler2, g)
    decoded2 = dec2.decode()
    assert decoded2.symbol_block == original_block, "Format Info 单错误后解码失败"
    print(f"  [OK] Format Info 单模块翻转后 BCH 纠错成功")


def test_e2e_multiple_frames():
    """模拟多帧编解码，验证帧间独立性。"""
    print("\n=== E2E 测试: 多帧独立性 ===")

    pal = ColorPalette(8)
    shp = ShapeSet(4)
    g = Grid(3)
    enc = SymbolEncoder(pal, shp, g)

    for frame_idx in range(5):
        block = [random.randint(0, enc.M - 1) for _ in range(enc.S)]
        frame = enc.encode(block)
        sampler = MockSampler(frame, pal)
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()
        assert decoded.symbol_block == block, f"帧 {frame_idx} 往返失败"
        print(f"  [OK] 帧{frame_idx}: S={enc.S}, M={enc.M}, 往返一致")

    print("[OK] 多帧独立性测试通过")


# ============================================================
# 真实图像 E2E 测试：encode → render PNG → load → decode
# ============================================================

import os
import tempfile
from PIL import Image
from symbol_layer import FrameRenderer, ImageSampler, OpenCVSampler

_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")


def test_real_image_e2e():
    """encode → 渲染 PNG → 读取 PNG → 解码 → 验证符号块一致
    覆盖不同等级、颜色档、图形档和模块尺寸（生成不同大小图像）。
    """
    print("\n=== 真实图像 E2E 测试: 不同尺寸 render PNG -> load -> decode ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # (level, n_colors, n_shapes, module_size)
    test_configs = [
        # 小尺寸
        (1, 2, 2, 4),    # 等级1, 2色+2图形, ms=4  → 116px
        (1, 4, 4, 5),    # 等级1, 4色+4图形, ms=5  → 145px
        (2, 4, 4, 4),    # 等级2, 4色+4图形, ms=4  → 148px
        # 中尺寸
        (1, 2, 2, 10),   # 等级1, 2色+2图形, ms=10 → 290px
        (1, 4, 4, 10),   # 等级1, 4色+4图形, ms=10 → 290px
        (2, 4, 4, 8),    # 等级2, 4色+4图形, ms=8  → 296px
        (2, 8, 4, 8),    # 等级2, 8色+4图形, ms=8  → 296px
        (3, 4, 4, 6),    # 等级3, 4色+4图形, ms=6  → 270px
        # 大尺寸
        (2, 4, 4, 16),   # 等级2, 4色+4图形, ms=16 → 592px
        (3, 8, 4, 12),   # 等级3, 8色+4图形, ms=12 → 540px
        (4, 4, 4, 10),   # 等级4, 4色+4图形, ms=10 → 530px
    ]

    for level, n_c, n_s, ms in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        # 渲染为 PNG
        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img_path = os.path.join(_OUTPUT_DIR, f"frame_L{level}_C{n_c}_S{n_s}_ms{ms}.png")
        img.save(img_path)

        # 验证图像尺寸
        expected_size = (g.N + 8) * ms
        assert img.size == (expected_size, expected_size), \
            f"图像尺寸 {img.size} != ({expected_size}, {expected_size})"

        # 从 PNG 读取并解码
        loaded_img = Image.open(img_path)
        sampler = ImageSampler(
            image=loaded_img,
            grid=g,
            palette=pal,
            shape_set=shp,
            module_size=ms,
            quiet_zone_size=4,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        # 验证符号块
        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        assert mismatches == 0, \
            f"等级{level} {n_c}色+{n_s}图形 ms={ms}: {mismatches}/{S} 个符号不匹配"

        # 验证 Format Info
        assert decoded.format_info.color_level_code == pal.code
        assert decoded.format_info.shape_level_code == shp.code

        print(f"  [OK] L{level} C{n_c} S{n_s} ms={ms}: {expected_size}x{expected_size}px, "
              f"S={S}, M={M}, 0 mismatches")

    print("[OK] 真实图像 E2E 往返测试通过 (11 种尺寸)")


def test_real_image_all_symbols():
    """真实图像全符号值遍历：确保每个符号值都能正确渲染和识别"""
    print("\n=== 真实图像 E2E: 全符号值遍历 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    for n_c in [2, 4, 8]:
        for n_s in [2, 4]:
            pal = ColorPalette(n_c)
            shp = ShapeSet(n_s)
            g = Grid(1)
            enc = SymbolEncoder(pal, shp, g)
            M = enc.M
            S = enc.S

            test_block = [0] * S
            for i in range(min(M, S)):
                test_block[i] = i

            frame = enc.encode(test_block)
            renderer = FrameRenderer(module_size=12, quiet_zone_size=4)
            img = renderer.render(frame)

            sampler = ImageSampler(
                image=img, grid=g, palette=pal, shape_set=shp,
                module_size=12, quiet_zone_size=4,
            )
            dec = SymbolDecoder(sampler, g)
            decoded = dec.decode()

            mismatches = sum(1 for a, b in zip(decoded.symbol_block, test_block) if a != b)
            assert mismatches == 0, \
                f"{n_c}色+{n_s}图形: {mismatches} 个符号不匹配"
            print(f"  [OK] {n_c}色+{n_s}图形: M={M}, 全符号值真实图像往返一致")

    print("[OK] 真实图像全符号值遍历通过")


def test_real_image_format_info_correction():
    """真实图像 Format Info 纠错：翻转图像中 Format Info 的像素，验证 BCH 纠错"""
    print("\n=== 真实图像 E2E: Format Info 像素翻转纠错 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    pal = ColorPalette(4)
    shp = ShapeSet(4)
    g = Grid(2)
    enc = SymbolEncoder(pal, shp, g)
    original_block = [random.randint(0, enc.M - 1) for _ in range(enc.S)]
    frame = enc.encode(original_block)

    ms = 10
    renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
    img = renderer.render(frame)

    # 翻转第一份 Format Info 第一个模块的全部像素
    fi_positions = g.format_info_positions
    c, r = fi_positions[0][0]
    px = (c + 4) * ms
    py = (r + 4) * ms

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # 取反色
    for dy in range(ms):
        for dx in range(ms):
            pixel = img.getpixel((px + dx, py + dy))
            inverted = tuple(255 - c for c in pixel)
            draw.point((px + dx, py + dy), fill=inverted)

    # 解码（应通过 BCH 纠错或第二份冗余恢复）
    sampler = ImageSampler(
        image=img, grid=g, palette=pal, shape_set=shp,
        module_size=ms, quiet_zone_size=4,
    )
    dec = SymbolDecoder(sampler, g)
    decoded = dec.decode()

    mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
    assert mismatches == 0, f"Format Info 像素翻转后解码失败: {mismatches} 个不匹配"
    print(f"  [OK] Format Info 像素翻转后 BCH 纠错成功, 0 mismatches")


def test_real_image_save_and_reload():
    """测试保存到磁盘再重新加载的完整流程"""
    print("\n=== 真实图像 E2E: 保存/重新加载 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    pal = ColorPalette(8)
    shp = ShapeSet(4)
    g = Grid(2)
    enc = SymbolEncoder(pal, shp, g)
    original_block = [random.randint(0, enc.M - 1) for _ in range(enc.S)]
    frame = enc.encode(original_block)

    ms = 8
    renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
    img = renderer.render(frame)

    # 保存
    path = os.path.join(_OUTPUT_DIR, "test_save_reload.png")
    img.save(path)
    assert os.path.exists(path), f"文件未创建: {path}"
    file_size = os.path.getsize(path)
    assert file_size > 0, "文件为空"

    # 重新加载
    reloaded = Image.open(path)
    sampler = ImageSampler(
        image=reloaded, grid=g, palette=pal, shape_set=shp,
        module_size=ms, quiet_zone_size=4,
    )
    dec = SymbolDecoder(sampler, g)
    decoded = dec.decode()

    mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
    assert mismatches == 0, f"保存/重新加载后解码失败: {mismatches} 个不匹配"
    print(f"  [OK] 保存/重新加载: {path}, {file_size} bytes, 0 mismatches")


def test_real_image_with_noise():
    """添加随机噪声后解码，验证鲁棒性"""
    print("\n=== 真实图像 E2E: 噪声干扰 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_configs = [
        (1, 4, 4, 12, 10),   # 等级1, 4色+4图形, ms=12, noise=10
        (1, 4, 4, 12, 20),   # 等级1, 4色+4图形, ms=12, noise=20
        (2, 4, 4, 10, 10),   # 等级2, 4色+4图形, ms=10, noise=10
        (2, 8, 4, 10, 8),    # 等级2, 8色+4图形, ms=10, noise=8
        (3, 4, 4, 8, 10),    # 等级3, 4色+4图形, ms=8, noise=10
    ]

    for level, n_c, n_s, ms, noise in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img_noisy = FrameRenderer.add_noise(img, intensity=noise, seed=42)

        img_path = os.path.join(_OUTPUT_DIR, f"noisy_L{level}_C{n_c}_S{n_s}_ms{ms}_n{noise}.png")
        img_noisy.save(img_path)

        sampler = ImageSampler(
            image=img_noisy, grid=g, palette=pal, shape_set=shp,
            module_size=ms, quiet_zone_size=4,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        total = S
        error_rate = mismatches / total

        print(f"  [{'OK' if mismatches == 0 else 'WARN'}] L{level} C{n_c} S{n_s} ms={ms} noise={noise}: "
              f"{mismatches}/{S} mismatches ({error_rate:.1%}), saved to {img_path}")

        # 噪声场景允许少量错误，但错误率不应过高
        assert error_rate < 0.05, f"噪声干扰后错误率过高: {error_rate:.1%}"

    print("[OK] 噪声干扰测试通过 (错误率均 < 5%)")


def test_real_image_with_blur():
    """添加高斯模糊后解码，验证鲁棒性"""
    print("\n=== 真实图像 E2E: 模糊干扰 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_configs = [
        (1, 4, 4, 12, 0.5),   # 轻微模糊
        (1, 4, 4, 12, 1.0),   # 中等模糊
        (2, 4, 4, 10, 0.8),   # 等级2 中等模糊
        (2, 8, 4, 10, 0.5),   # 等级2 8色 轻微模糊
        (3, 4, 4, 8, 0.6),    # 等级3 模糊
    ]

    for level, n_c, n_s, ms, blur_r in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img_blurred = FrameRenderer.add_blur(img, radius=blur_r)

        img_path = os.path.join(_OUTPUT_DIR, f"blur_L{level}_C{n_c}_S{n_s}_ms{ms}_r{blur_r}.png")
        img_blurred.save(img_path)

        sampler = ImageSampler(
            image=img_blurred, grid=g, palette=pal, shape_set=shp,
            module_size=ms, quiet_zone_size=4,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        error_rate = mismatches / S

        print(f"  [{'OK' if mismatches == 0 else 'WARN'}] L{level} C{n_c} S{n_s} ms={ms} blur={blur_r}: "
              f"{mismatches}/{S} mismatches ({error_rate:.1%}), saved to {img_path}")

        assert error_rate < 0.05, f"模糊干扰后错误率过高: {error_rate:.1%}"

    print("[OK] 模糊干扰测试通过 (错误率均 < 5%)")


def test_real_image_combined_distortion():
    """同时添加噪声 + 模糊，验证综合干扰下的鲁棒性"""
    print("\n=== 真实图像 E2E: 噪声+模糊组合干扰 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_configs = [
        (1, 4, 4, 12, 8, 0.5),    # 等级1, 轻噪声+轻模糊
        (2, 4, 4, 10, 10, 0.8),   # 等级2, 中噪声+中模糊
        (2, 8, 4, 10, 6, 0.5),    # 等级2 8色, 轻噪声+轻模糊
    ]

    for level, n_c, n_s, ms, noise, blur_r in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img_distorted = FrameRenderer.add_noise(img, intensity=noise, seed=42)
        img_distorted = FrameRenderer.add_blur(img_distorted, radius=blur_r)

        img_path = os.path.join(_OUTPUT_DIR, f"distort_L{level}_C{n_c}_S{n_s}_ms{ms}_n{noise}_b{blur_r}.png")
        img_distorted.save(img_path)

        sampler = ImageSampler(
            image=img_distorted, grid=g, palette=pal, shape_set=shp,
            module_size=ms, quiet_zone_size=4,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        error_rate = mismatches / S

        print(f"  [{'OK' if mismatches == 0 else 'WARN'}] L{level} C{n_c} S{n_s} ms={ms} "
              f"noise={noise} blur={blur_r}: {mismatches}/{S} mismatches ({error_rate:.1%})")

        assert error_rate < 0.10, f"组合干扰后错误率过高: {error_rate:.1%}"

    print("[OK] 组合干扰测试通过 (错误率均 < 10%)")


def test_real_image_with_affine():
    """轻微仿射变换（旋转+缩放+平移）后解码，验证鲁棒性"""
    print("\n=== 真实图像 E2E: 仿射变换 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_configs = [
        # (level, n_c, n_s, ms, rotation_deg, scale, translate_x, translate_y)
        (1, 4, 4, 12, 1.0, 0.98, 2.0, 1.0),    # 轻微旋转+缩放+平移
        (1, 4, 4, 12, 2.0, 0.96, 3.0, 2.0),    # 稍大旋转+缩放
        (2, 4, 4, 10, 1.5, 0.97, 2.0, 1.0),    # 等级2
        (2, 8, 4, 10, 1.0, 0.98, 1.0, 1.0),    # 等级2 8色
        (3, 4, 4, 8, 1.0, 0.98, 1.0, 0.0),     # 等级3
    ]

    for level, n_c, n_s, ms, rot, scl, tx, ty in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img_affine = FrameRenderer.add_affine(img, rotation_deg=rot, scale=scl,
                                               translate_x=tx, translate_y=ty)

        img_path = os.path.join(_OUTPUT_DIR,
            f"affine_L{level}_C{n_c}_S{n_s}_ms{ms}_r{rot}_s{scl}.png")
        img_affine.save(img_path)

        # 使用 OpenCVSampler 自动检测 Finder Pattern 校正仿射变换
        sampler = OpenCVSampler(
            image=img_affine, grid=g, palette=pal, shape_set=shp,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        error_rate = mismatches / S

        status = "OK" if error_rate < 0.05 else "WARN"
        print(f"  [{status}] L{level} C{n_c} S{n_s} ms={ms} rot={rot} scale={scl} "
              f"tx={tx} ty={ty}: {mismatches}/{S} ({error_rate:.1%}), saved")

        assert error_rate < 0.10, f"仿射变换后错误率过高: {error_rate:.1%}"

    print("[OK] 仿射变换测试通过 (错误率均 < 10%)")


def test_real_image_with_lens_distortion():
    """镜头畸变（桶形/枕形）后解码，验证鲁棒性"""
    print("\n=== 真实图像 E2E: 镜头畸变 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_configs = [
        # (level, n_c, n_s, ms, k1, k2)
        (1, 4, 4, 12, 0.005, 0.0),   # 轻微桶形畸变
        (1, 4, 4, 12, 0.01, 0.0),    # 中等桶形畸变
        (1, 4, 4, 12, -0.01, 0.0),   # 轻微枕形畸变
        (2, 4, 4, 14, 0.005, 0.0),   # 等级2 桶形
        (2, 8, 4, 14, 0.005, 0.0),   # 等级2 8色 桶形
        (3, 4, 4, 12, 0.003, 0.0),   # 等级3 桶形
    ]

    for level, n_c, n_s, ms, k1, k2 in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img_lens = FrameRenderer.add_lens_distortion(img, k1=k1, k2=k2)

        distortion_type = "barrel" if k1 > 0 else "pincushion" if k1 < 0 else "none"
        img_path = os.path.join(_OUTPUT_DIR,
            f"lens_{distortion_type}_L{level}_C{n_c}_S{n_s}_ms{ms}_k{k1}.png")
        img_lens.save(img_path)

        # 使用 OpenCVSampler 自动检测 Finder Pattern 校正镜头畸变
        sampler = OpenCVSampler(
            image=img_lens, grid=g, palette=pal, shape_set=shp,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        error_rate = mismatches / S

        status = "OK" if error_rate < 0.05 else "WARN"
        print(f"  [{status}] L{level} C{n_c} S{n_s} ms={ms} k1={k1} k2={k2} "
              f"({distortion_type}): {mismatches}/{S} ({error_rate:.1%}), saved")

        assert error_rate < 0.10, f"镜头畸变后错误率过高: {error_rate:.1%}"

    print("[OK] 镜头畸变测试通过 (错误率均 < 10%)")


def test_real_image_with_perspective():
    """俯拍透视畸变后解码，验证透视变换校正能力"""
    print("\n=== 真实图像 E2E: 俯拍透视畸变 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_configs = [
        # (level, n_c, n_s, ms, tilt_x, tilt_y)
        (1, 4, 4, 12, 0.05, 0.0),    # 轻度倾斜
        (1, 4, 4, 12, 0.10, 0.0),    # 中度倾斜
        (1, 4, 4, 12, 0.15, 0.0),    # 较大倾斜
        (1, 4, 4, 12, 0.10, 0.05),   # 双轴倾斜
        (1, 4, 4, 12, 0.15, 0.10),   # 双轴较大倾斜
        (2, 4, 4, 12, 0.10, 0.0),    # 等级2 中度倾斜
        (2, 4, 4, 12, 0.15, 0.10),   # 等级2 双轴倾斜
        (2, 8, 4, 12, 0.10, 0.05),   # 等级2 8色 双轴倾斜
    ]

    for level, n_c, n_s, ms, tilt_x, tilt_y in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img_persp = FrameRenderer.add_perspective(img, tilt_x=tilt_x, tilt_y=tilt_y)

        img_path = os.path.join(_OUTPUT_DIR,
            f"persp_L{level}_C{n_c}_S{n_s}_ms{ms}_tx{tilt_x}_ty{tilt_y}.png")
        img_persp.save(img_path)

        sampler = OpenCVSampler(
            image=img_persp, grid=g, palette=pal, shape_set=shp,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        error_rate = mismatches / S

        status = "OK" if error_rate < 0.05 else "WARN"
        print(f"  [{status}] L{level} C{n_c} S{n_s} ms={ms} tx={tilt_x} ty={tilt_y}: "
              f"{mismatches}/{S} ({error_rate:.1%}), saved")

        assert error_rate < 0.10, f"透视畸变后错误率过高: {error_rate:.1%}"

    print("[OK] 俯拍透视畸变测试通过 (错误率均 < 10%)")


def test_real_image_with_offset():
    """码图偏离图像中心后解码，验证 detect() 对非居中码图的适应性"""
    print("\n=== 真实图像 E2E: 偏离中心 ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_configs = [
        # (level, n_c, n_s, ms, offset_x, offset_y, margin)
        (1, 4, 4, 12, 0.1, 0.1, 0.3),     # 轻度偏移
        (1, 4, 4, 12, 0.3, 0.2, 0.3),     # 中度偏移
        (1, 4, 4, 12, -0.2, 0.3, 0.3),    # 偏左下
        (1, 4, 4, 12, 0.5, -0.3, 0.3),    # 大幅偏移
        (1, 4, 4, 12, 0.0, 0.5, 0.3),     # 偏下
        (2, 4, 4, 12, 0.3, 0.2, 0.3),     # 等级2 中度偏移
        (2, 8, 4, 12, -0.2, 0.3, 0.3),    # 等级2 8色 偏左下
        (2, 4, 4, 12, 0.5, -0.3, 0.4),    # 等级2 大幅偏移
    ]

    for level, n_c, n_s, ms, ox, oy, margin in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img_offset = FrameRenderer.add_offset(img, offset_x=ox, offset_y=oy, margin=margin)

        img_path = os.path.join(_OUTPUT_DIR,
            f"offset_L{level}_C{n_c}_S{n_s}_ms{ms}_ox{ox}_oy{oy}.png")
        img_offset.save(img_path)

        sampler = OpenCVSampler(
            image=img_offset, grid=g, palette=pal, shape_set=shp,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        error_rate = mismatches / S

        status = "OK" if error_rate < 0.05 else "WARN"
        print(f"  [{status}] L{level} C{n_c} S{n_s} ms={ms} ox={ox} oy={oy}: "
              f"{mismatches}/{S} ({error_rate:.1%}), img={img_offset.size}, saved")

        assert error_rate < 0.10, f"偏离中心后错误率过高: {error_rate:.1%}"

    print("[OK] 偏离中心测试通过 (错误率均 < 10%)")


def test_real_image_full_distortion():
    """全干扰组合：噪声 + 模糊 + 仿射变换 + 镜头畸变"""
    print("\n=== 真实图像 E2E: 全干扰组合 (噪声+模糊+仿射+畸变) ===")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_configs = [
        # (level, n_c, n_s, ms, noise, blur, rot, scale, k1)
        (1, 4, 4, 12, 5, 0.3, 0.3, 0.99, 0.003),   # 轻度全干扰
        (2, 4, 4, 14, 5, 0.3, 0.3, 0.99, 0.003),   # 等级2 轻度
        (2, 8, 4, 14, 3, 0.2, 0.2, 0.99, 0.002),   # 等级2 8色 轻度
    ]

    for level, n_c, n_s, ms, noise, blur, rot, scl, k1 in test_configs:
        g = Grid(level)
        pal = ColorPalette(n_c)
        shp = ShapeSet(n_s)
        enc = SymbolEncoder(pal, shp, g)
        M = enc.M
        S = enc.S

        original_block = [random.randint(0, M - 1) for _ in range(S)]
        frame = enc.encode(original_block)

        renderer = FrameRenderer(module_size=ms, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_noise(img, intensity=noise, seed=42)
        img = FrameRenderer.add_blur(img, radius=blur)
        img = FrameRenderer.add_affine(img, rotation_deg=rot, scale=scl)
        img = FrameRenderer.add_lens_distortion(img, k1=k1)

        img_path = os.path.join(_OUTPUT_DIR,
            f"full_distort_L{level}_C{n_c}_S{n_s}_ms{ms}_n{noise}_b{blur}_r{rot}_k{k1}.png")
        img.save(img_path)

        # 使用 OpenCVSampler 自动检测 Finder Pattern 校正全干扰
        sampler = OpenCVSampler(
            image=img, grid=g, palette=pal, shape_set=shp,
        )
        dec = SymbolDecoder(sampler, g)
        decoded = dec.decode()

        mismatches = sum(1 for a, b in zip(decoded.symbol_block, original_block) if a != b)
        error_rate = mismatches / S

        status = "OK" if error_rate < 0.05 else "WARN"
        print(f"  [{status}] L{level} C{n_c} S{n_s} ms={ms} noise={noise} blur={blur} "
              f"rot={rot} scale={scl} k1={k1}: {mismatches}/{S} ({error_rate:.1%}), saved")

        assert error_rate < 0.15, f"全干扰组合后错误率过高: {error_rate:.1%}"

    print("[OK] 全干扰组合测试通过 (错误率均 < 15%)")


if __name__ == "__main__":
    main()
    test_e2e()
    test_e2e_all_symbols()
    test_e2e_format_info_error_correction()
    test_e2e_multiple_frames()
    test_real_image_e2e()
    test_real_image_all_symbols()
    test_real_image_format_info_correction()
    test_real_image_save_and_reload()
    test_real_image_with_noise()
    test_real_image_with_blur()
    test_real_image_combined_distortion()
    test_real_image_with_affine()
    test_real_image_with_lens_distortion()
    test_real_image_with_perspective()
    test_real_image_with_offset()
    test_real_image_full_distortion()
    print("\n" + "=" * 60)
    print("全部测试通过 (单元 + E2E + 真实图像 + 噪声/模糊/仿射/畸变/全干扰)")
    print("=" * 60)
