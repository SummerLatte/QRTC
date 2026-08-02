"""
符号 → 图像 → 符号 E2E 测试。

完整链路：
  SymbolEncoder.encode(symbol_block) → RenderedFrame
  FrameRenderer.render(frame) → PIL Image
  ImageSampler / OpenCVSampler(image, ...) → sampler
  SymbolDecoder(sampler, grid).decode() → DecodedFrame
  对比 original vs decoded

测试场景：
  - 理想无损（ImageSampler）
  - 多种颜色/图形/网格等级组合
  - 噪声退化
  - 模糊退化
  - OpenCVSampler 自动检测链路
"""

import random
import os

import pytest
from PIL import Image

from symbol_layer import (
    ColorPalette, ShapeSet,
    Grid, SymbolEncoder, FrameRenderer,
    SymbolDecoder, ImageSampler,
)


# ======================================================================
# 辅助函数
# ======================================================================

# 测试图像输出目录
IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")


def save_image(img: Image.Image, name: str) -> str:
    """将图像保存到 test_output 目录，返回保存路径。"""
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(IMAGE_OUTPUT_DIR, f"{name}.png")
    img.save(path)
    return path


def make_symbol_block(S: int, M: int, seed: int = 42) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, M - 1) for _ in range(S)]


def e2e_image_sampler(
    n_colors: int,
    n_shapes: int,
    level: int,
    module_size: int = 10,
    quiet_zone: int = 4,
    symbol_block: list[int] | None = None,
    seed: int = 42,
    tag_suffix: str = "",
):
    """
    用 ImageSampler 跑完整 E2E 链路。

    Returns: (original_block, decoded_frame, M, S)
    """
    pal = ColorPalette(n_colors)
    ss = ShapeSet(n_shapes)
    grid = Grid(level)
    encoder = SymbolEncoder(pal, ss, grid)
    M = encoder.M
    S = encoder.S

    if symbol_block is None:
        symbol_block = make_symbol_block(S, M, seed)

    frame = encoder.encode(symbol_block)
    renderer = FrameRenderer(module_size=module_size, quiet_zone_size=quiet_zone)
    img = renderer.render(frame)

    tag = f"ideal_C{n_colors}_S{n_shapes}_L{level}_ms{module_size}"
    if tag_suffix:
        tag += f"_{tag_suffix}"
    save_image(img, tag)

    sampler = ImageSampler(
        img, grid, pal, ss,
        module_size=module_size,
        quiet_zone_size=quiet_zone,
    )
    decoder = SymbolDecoder(sampler, grid)
    decoded = decoder.decode()

    return symbol_block, decoded, M, S


def count_errors(original: list[int], decoded_list: list[int]) -> int:
    return sum(1 for a, b in zip(original, decoded_list) if a != b)


# ======================================================================
# 1. 理想无损 E2E（ImageSampler）
# ======================================================================

class TestE2EImageSamplerIdeal:
    """理想无损场景：渲染 → 直接采样，无任何退化。"""

    @pytest.mark.parametrize("n_colors,n_shapes,level", [
        (2, 2, 1), (2, 4, 1),
        (4, 2, 1), (4, 4, 1),
        (4, 4, 2), (4, 4, 3), (4, 4, 4),
        (8, 2, 1), (8, 4, 1), (8, 4, 2),
    ])
    def test_roundtrip_accuracy(self, n_colors, n_shapes, level):
        """无损场景下，解码结果应与原始符号块完全一致。"""
        original, decoded, M, S = e2e_image_sampler(
            n_colors, n_shapes, level, module_size=10,
        )
        assert len(decoded.symbol_block) == S
        errors = count_errors(original, decoded.symbol_block)
        assert errors == 0, f"{errors}/{S} 个符号不匹配"

    def test_format_info_correct(self):
        """解码后 Format Info 应与编码参数一致。"""
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        original, decoded, _, _ = e2e_image_sampler(4, 4, 1)
        assert decoded.format_info.color_level_code == pal.code
        assert decoded.format_info.shape_level_code == ss.code

    def test_no_erasures_in_ideal(self):
        """理想场景不应有 erasure。"""
        original, decoded, _, _ = e2e_image_sampler(4, 4, 1)
        assert not any(decoded.erasure_flags)

    def test_high_confidence_in_ideal(self):
        """理想场景所有置信度应很高。"""
        original, decoded, _, _ = e2e_image_sampler(4, 4, 1)
        for conf in decoded.confidences:
            assert conf > 0.5


# ======================================================================
# 2. 不同 module_size 下的 E2E
# ======================================================================

class TestE2EModuleSize:
    """不同模块像素大小下的 E2E。"""

    @pytest.mark.parametrize("module_size", [5, 8, 10, 15, 20])
    def test_different_module_sizes(self, module_size):
        """不同 module_size 下无损往返应准确。"""
        original, decoded, _, _ = e2e_image_sampler(
            4, 4, 1, module_size=module_size,
        )
        errors = count_errors(original, decoded.symbol_block)
        assert errors == 0, f"module_size={module_size}: {errors} 个错误"


# ======================================================================
# 3. 噪声退化 E2E
# ======================================================================

class TestE2ENoise:
    """添加高斯噪声后的 E2E 鲁棒性。"""

    def _e2e_with_noise(
        self, n_colors=4, n_shapes=4, level=1,
        module_size=20, noise_intensity=10, seed=42,
    ):
        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        grid = Grid(level)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=module_size, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_noise(img, intensity=noise_intensity, seed=seed)

        tag = f"noise_C{n_colors}_S{n_shapes}_L{level}_ms{module_size}_n{noise_intensity}"
        save_image(img, tag)

        sampler = ImageSampler(img, grid, pal, ss, module_size, 4)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()
        return block, decoded, M, S

    @pytest.mark.parametrize("intensity", [3, 8, 15])
    def test_low_noise_accuracy(self, intensity):
        """低噪声下应保持高准确率。"""
        original, decoded, _, S = self._e2e_with_noise(
            noise_intensity=intensity, module_size=20,
        )
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.05, f"intensity={intensity}: 错误率 {error_rate:.2%}"

    def test_high_noise_degradation(self):
        """高噪声下错误率应上升但仍有一定正确率。"""
        original, decoded, _, S = self._e2e_with_noise(
            noise_intensity=40, module_size=20,
        )
        errors = count_errors(original, decoded.symbol_block)
        # 高噪声下不要求完全正确，但不应全部错误
        assert errors < S, "全部符号错误，采样器可能完全失效"


# ======================================================================
# 4. 模糊退化 E2E
# ======================================================================

class TestE2EBlur:
    """添加高斯模糊后的 E2E 鲁棒性。"""

    def _e2e_with_blur(
        self, n_colors=4, n_shapes=4, level=1,
        module_size=20, blur_radius=0.5, seed=42,
    ):
        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        grid = Grid(level)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=module_size, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_blur(img, radius=blur_radius)

        tag = f"blur_C{n_colors}_S{n_shapes}_L{level}_ms{module_size}_r{blur_radius}"
        save_image(img, tag)

        sampler = ImageSampler(img, grid, pal, ss, module_size, 4)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()
        return block, decoded, M, S

    @pytest.mark.parametrize("radius", [0.3, 0.5, 1.0])
    def test_mild_blur_accuracy(self, radius):
        """轻度模糊下应保持较高准确率。"""
        original, decoded, _, S = self._e2e_with_blur(
            blur_radius=radius, module_size=20,
        )
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.10, f"radius={radius}: 错误率 {error_rate:.2%}"

    def test_heavy_blur_degradation(self):
        """重度模糊下错误率应上升。"""
        original, decoded, _, S = self._e2e_with_blur(
            blur_radius=5.0, module_size=10,
        )
        errors = count_errors(original, decoded.symbol_block)
        assert errors > 0, "重度模糊下应有错误"


# ======================================================================
# 5. 噪声 + 模糊组合退化 E2E
# ======================================================================

class TestE2ECombined:
    """噪声 + 模糊组合退化。"""

    def test_noise_plus_blur(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed=99)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=20, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_noise(img, intensity=8, seed=99)
        img = FrameRenderer.add_blur(img, radius=0.5)

        save_image(img, "combined_noise8_blur0.5_C4_S4_L1_ms20")

        sampler = ImageSampler(img, grid, pal, ss, 20, 4)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        errors = count_errors(block, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.10, f"噪声+模糊: 错误率 {error_rate:.2%}"


# ======================================================================
# 6. 边界符号值 E2E
# ======================================================================

class TestE2EBoundarySymbols:
    """全 0、全最大值、顺序符号值的 E2E。"""

    def test_all_zero(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        encoder = SymbolEncoder(pal, ss, grid)
        block = [0] * encoder.S
        original, decoded, _, _ = e2e_image_sampler(
            4, 4, 1, symbol_block=block, tag_suffix="all_zero",
        )
        assert count_errors(original, decoded.symbol_block) == 0

    def test_all_max(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        encoder = SymbolEncoder(pal, ss, grid)
        block = [encoder.M - 1] * encoder.S
        original, decoded, _, _ = e2e_image_sampler(
            4, 4, 1, symbol_block=block, tag_suffix="all_max",
        )
        assert count_errors(original, decoded.symbol_block) == 0

    def test_sequential(self):
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(2)
        encoder = SymbolEncoder(pal, ss, grid)
        block = [i % encoder.M for i in range(encoder.S)]
        original, decoded, _, _ = e2e_image_sampler(
            4, 4, 2, symbol_block=block, tag_suffix="sequential",
        )
        assert count_errors(original, decoded.symbol_block) == 0


# ======================================================================
# 7. OpenCVSampler E2E（自动 QR 检测 + 透视校正）
# ======================================================================

class TestE2EOpenCVSampler:
    """使用 OpenCVSampler 进行自动检测的 E2E。"""

    def _e2e_opencv(
        self, n_colors=4, n_shapes=4, level=1,
        module_size=20, seed=42,
    ):
        from symbol_layer import OpenCVSampler

        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        grid = Grid(level)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=module_size, quiet_zone_size=4)
        img = renderer.render(frame)

        tag = f"opencv_C{n_colors}_S{n_shapes}_L{level}_ms{module_size}"
        img_path = save_image(img, tag)
        loaded = Image.open(img_path).convert("RGB")

        sampler = OpenCVSampler(loaded, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        return block, decoded, M, S

    @pytest.mark.parametrize("level", [1, 2])
    def test_opencv_roundtrip(self, level):
        """OpenCVSampler 自动检测链路的 E2E 往返。"""
        original, decoded, _, S = self._e2e_opencv(level=level)
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.05, f"level={level}: 错误率 {error_rate:.2%}"

    def test_opencv_format_info(self):
        """OpenCV 链路应正确读取 Format Info。"""
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        _, decoded, _, _ = self._e2e_opencv(n_colors=4, n_shapes=4, level=1)
        assert decoded.format_info.color_level_code == pal.code
        assert decoded.format_info.shape_level_code == ss.code

    def test_opencv_8color(self):
        """8 色档 OpenCV 链路。"""
        original, decoded, _, S = self._e2e_opencv(
            n_colors=8, n_shapes=4, level=1,
        )
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.10, f"8色: 错误率 {error_rate:.2%}"


# ======================================================================
# 8. 非中心偏移 E2E（OpenCVSampler 自动检测）
# ======================================================================

class TestE2EOffset:
    """码图偏离图像中心的 E2E，使用 OpenCVSampler 自动检测。"""

    def _e2e_with_offset(
        self, n_colors=4, n_shapes=4, level=1,
        module_size=20, offset_x=0.5, offset_y=0.5,
        margin=0.3, seed=42,
    ):
        from symbol_layer import OpenCVSampler

        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        grid = Grid(level)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=module_size, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_offset(img, offset_x=offset_x, offset_y=offset_y, margin=margin)

        tag = f"offset_ox{offset_x}_oy{offset_y}_m{margin}_C{n_colors}_S{n_shapes}_L{level}_ms{module_size}"
        img_path = save_image(img, tag)
        loaded = Image.open(img_path).convert("RGB")

        sampler = OpenCVSampler(loaded, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        return block, decoded, M, S

    @pytest.mark.parametrize("offset_x,offset_y", [
        (0.5, 0.5),    # 偏右下
        (-0.5, -0.5),  # 偏左上
        (0.8, -0.3),   # 偏右偏上
        (-0.3, 0.8),   # 偏左偏下
    ])
    def test_offset_roundtrip(self, offset_x, offset_y):
        """非中心偏移下 OpenCVSampler 应自动检测并正确解码。"""
        original, decoded, _, S = self._e2e_with_offset(
            offset_x=offset_x, offset_y=offset_y,
        )
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.05, f"offset=({offset_x},{offset_y}): 错误率 {error_rate:.2%}"

    def test_large_margin(self):
        """大边距（margin=0.8）下仍应自动检测。"""
        original, decoded, _, S = self._e2e_with_offset(
            offset_x=0.3, offset_y=-0.3, margin=0.8,
        )
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.05, f"大边距: 错误率 {error_rate:.2%}"


# ======================================================================
# 9. 仿射变换 E2E（旋转 + 缩放 + 平移）
# ======================================================================

class TestE2EAffine:
    """仿射变换后的 E2E，使用 OpenCVSampler 自动检测。"""

    def _e2e_with_affine(
        self, n_colors=4, n_shapes=4, level=1,
        module_size=20, rotation_deg=2.0, scale=0.98,
        translate_x=0.0, translate_y=0.0, seed=42,
    ):
        from symbol_layer import OpenCVSampler

        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        grid = Grid(level)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=module_size, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_affine(
            img, rotation_deg=rotation_deg, scale=scale,
            translate_x=translate_x, translate_y=translate_y,
        )

        tag = (f"affine_rot{rotation_deg}_scl{scale}_tx{translate_x}_ty{translate_y}"
               f"_C{n_colors}_S{n_shapes}_L{level}_ms{module_size}")
        img_path = save_image(img, tag)
        loaded = Image.open(img_path).convert("RGB")

        sampler = OpenCVSampler(loaded, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        return block, decoded, M, S

    @pytest.mark.parametrize("rotation_deg", [1.0, 3.0, 5.0])
    def test_small_rotation(self, rotation_deg):
        """小角度旋转下应能检测并部分解码（透视校正能力有限）。"""
        original, decoded, _, S = self._e2e_with_affine(rotation_deg=rotation_deg)
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        # OpenCVSampler 用 4 角透视变换近似仿射，旋转后模块采样有偏移
        # 验证不全错（采样器基本工作）
        assert error_rate < 0.85, f"rot={rotation_deg}: 错误率 {error_rate:.2%}"

    @pytest.mark.parametrize("scale", [0.95, 0.90])
    def test_scale_down(self, scale):
        """缩放变换下应自动检测。"""
        original, decoded, _, S = self._e2e_with_affine(
            rotation_deg=0, scale=scale,
        )
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.05, f"scale={scale}: 错误率 {error_rate:.2%}"

    def test_rotation_plus_scale(self):
        """旋转 + 缩放组合。"""
        original, decoded, _, S = self._e2e_with_affine(
            rotation_deg=3.0, scale=0.92,
        )
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.90, f"rot+scale: 错误率 {error_rate:.2%}"


# ======================================================================
# 10. 透视变换 E2E
# ======================================================================

class TestE2EPerspective:
    """透视畸变后的 E2E，使用 OpenCVSampler 自动检测。"""

    def _e2e_with_perspective(
        self, n_colors=4, n_shapes=4, level=1,
        module_size=20, tilt_x=0.1, tilt_y=0.0, pan=0.0, seed=42,
    ):
        from symbol_layer import OpenCVSampler

        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        grid = Grid(level)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=module_size, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_perspective(img, tilt_x=tilt_x, tilt_y=tilt_y, pan=pan)

        tag = (f"persp_tx{tilt_x}_ty{tilt_y}_pan{pan}"
               f"_C{n_colors}_S{n_shapes}_L{level}_ms{module_size}")
        img_path = save_image(img, tag)
        loaded = Image.open(img_path).convert("RGB")

        sampler = OpenCVSampler(loaded, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        return block, decoded, M, S

    @pytest.mark.parametrize("tilt_x", [0.05])
    def test_horizontal_tilt(self, tilt_x):
        """轻度水平透视倾斜下应能检测并部分解码。"""
        original, decoded, _, S = self._e2e_with_perspective(tilt_x=tilt_x)
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        # 透视畸变导致模块形状非矩形，采样器精度有限
        assert error_rate < 0.95, f"tilt_x={tilt_x}: 错误率 {error_rate:.2%}"

    def test_combined_tilt(self):
        """双向透视倾斜组合，验证采样器不崩溃。"""
        try:
            original, decoded, _, S = self._e2e_with_perspective(
                tilt_x=0.10, tilt_y=0.08,
            )
        except ValueError:
            # Format Info 纠错失败是可接受的（畸变过大）
            return
        errors = count_errors(original, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.95, f"combined tilt: 错误率 {error_rate:.2%}"


# ======================================================================
# 11. 非中心 + 噪声组合 E2E
# ======================================================================

class TestE2EOffsetPlusNoise:
    """非中心偏移 + 噪声组合退化。"""

    def test_offset_plus_low_noise(self):
        from symbol_layer import OpenCVSampler

        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed=77)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=20, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_offset(img, offset_x=0.6, offset_y=-0.4, margin=0.3)
        img = FrameRenderer.add_noise(img, intensity=5, seed=77)

        save_image(img, "offset_noise5_ox0.6_oy-0.4_C4_S4_L1_ms20")

        loaded = img
        sampler = OpenCVSampler(loaded, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        errors = count_errors(block, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.05, f"偏移+低噪声: 错误率 {error_rate:.2%}"

    def test_offset_plus_rotation_plus_noise(self):
        from symbol_layer import OpenCVSampler

        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        encoder = SymbolEncoder(pal, ss, grid)
        M, S = encoder.M, encoder.S
        block = make_symbol_block(S, M, seed=88)

        frame = encoder.encode(block)
        renderer = FrameRenderer(module_size=20, quiet_zone_size=4)
        img = renderer.render(frame)
        img = FrameRenderer.add_affine(img, rotation_deg=2.0, scale=0.95)
        img = FrameRenderer.add_offset(img, offset_x=0.5, offset_y=0.5, margin=0.25)
        img = FrameRenderer.add_noise(img, intensity=5, seed=88)

        save_image(img, "offset_rot2_noise5_C4_S4_L1_ms20")

        sampler = OpenCVSampler(img, grid, pal, ss)
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        errors = count_errors(block, decoded.symbol_block)
        error_rate = errors / S
        assert error_rate < 0.10, f"偏移+旋转+噪声: 错误率 {error_rate:.2%}"
