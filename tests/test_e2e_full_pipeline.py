"""
全链路 E2E 测试：应用层 → 传输层 → 数据层 → 符号层 → 图像 → 解码全链路返回。

完整编码链路：
  data → app_encode → content
  content → TransportCodec.encode → frames (L_max bytes each)
  each frame → DataCodec.encode → symbol block (S symbols)
  each symbol block → SymbolEncoder → RenderedFrame → FrameRenderer → PIL Image

完整解码链路：
  Image → ImageSampler → SymbolDecoder → DecodedFrame (symbols + erasures)
  symbols + erasures → DataCodec.decode → frame bytes (or None if RS/CRC fail)
  collected frames → TransportCodec.decode → content
  content → app_decode → AppMessage

测试场景：
  - 理想无损全链路往返
  - 多种颜色/图形/网格等级组合
  - 小文件 / 大文件
  - 文本消息
  - 帧丢失容错（LT 喷泉码）
  - 噪声退化下的全链路
  - 多 RS 块场景
"""

import os
import sys
import random
import struct

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application_layer import (
    app_encode, app_decode,
    AppMessage, VERSION, CONTENT_TYPE_FILE, CONTENT_TYPE_TEXT,
)
from transport_layer import TransportCodec
from data_layer import DataCodec
from symbol_layer import (
    ColorPalette, ShapeSet, Grid, SymbolEncoder, FrameRenderer,
    SymbolDecoder, ImageSampler,
)


# ======================================================================
# 辅助函数
# ======================================================================

IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")


def save_image(img: Image.Image, name: str) -> str:
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(IMAGE_OUTPUT_DIR, f"{name}.png")
    img.save(path)
    return path


def make_pipeline(n_colors=4, n_shapes=4, level=1, module_size=10, quiet_zone=4):
    """构建全链路所需的编码器/解码器集合。

    Returns: (pal, ss, grid, sym_encoder, data_codec, transport_codec, renderer_params)
    """
    pal = ColorPalette(n_colors)
    ss = ShapeSet(n_shapes)
    grid = Grid(level)
    sym_encoder = SymbolEncoder(pal, ss, grid)
    S = sym_encoder.S
    M = sym_encoder.M
    data_codec = DataCodec(S, M)
    L_max = data_codec.L_max
    transport_codec = TransportCodec(L_max)
    return pal, ss, grid, sym_encoder, data_codec, transport_codec


def full_encode(data: bytes, content_type: int, filename: str,
                n_colors, n_shapes, level,
                num_frames, module_size=10, quiet_zone=4, seed=42):
    """全链路编码：data → content → frames → symbol blocks → images。

    Returns: (images, transport_codec, data_codec, pal, ss, grid, content, total_length)
    """
    pal = ColorPalette(n_colors)
    ss = ShapeSet(n_shapes)
    grid = Grid(level)
    sym_encoder = SymbolEncoder(pal, ss, grid)
    S, M = sym_encoder.S, sym_encoder.M
    data_codec = DataCodec(S, M)
    L_max = data_codec.L_max
    transport_codec = TransportCodec(L_max)

    # 1. 应用层编码
    content = app_encode(data, content_type=content_type, filename=filename)
    total_length = len(content)

    # 2. 传输层编码
    frames = transport_codec.encode(content, num_frames)

    # 3. 数据层编码 + 符号层编码 + 渲染
    renderer = FrameRenderer(module_size=module_size, quiet_zone_size=quiet_zone)
    images = []
    for i, frame in enumerate(frames):
        assert len(frame) == L_max
        symbols = data_codec.encode(frame)
        assert len(symbols) == S
        rendered = sym_encoder.encode(symbols)
        img = renderer.render(rendered)
        images.append(img)

    return images, transport_codec, data_codec, pal, ss, grid, content, total_length


def full_decode(images, transport_codec, data_codec, pal, ss, grid,
                module_size=10, quiet_zone=4):
    """全链路解码：images → symbols → frames → content → AppMessage。

    Returns: (content, AppMessage or None)
    """
    # 4. 符号层解码 + 数据层解码
    decoded_frames = []
    for i, img in enumerate(images):
        sampler = ImageSampler(
            img, grid, pal, ss,
            module_size=module_size,
            quiet_zone_size=quiet_zone,
        )
        decoder = SymbolDecoder(sampler, grid)
        decoded = decoder.decode()

        # 数据层解码（含 RS 纠错 + CRC 校验）
        frame_bytes = data_codec.decode(
            decoded.symbol_block,
            erasure_flags=decoded.erasure_flags,
        )
        if frame_bytes is not None:
            decoded_frames.append(frame_bytes)

    # 5. 传输层解码
    content = transport_codec.decode(decoded_frames)
    if content is None:
        return None, None

    # 6. 应用层解码
    msg = app_decode(content)
    return content, msg


def run_full_e2e(
    data: bytes, content_type: int, filename: str,
    n_colors=4, n_shapes=4, level=1,
    num_frames=None, module_size=10, quiet_zone=4, seed=42,
):
    """运行完整 e2e 链路并返回 (content, AppMessage, total_length, num_frames_used)。"""
    # 先算 L_max 确定 K
    pal = ColorPalette(n_colors)
    ss = ShapeSet(n_shapes)
    grid = Grid(level)
    sym_encoder = SymbolEncoder(pal, ss, grid)
    S, M = sym_encoder.S, sym_encoder.M
    data_codec = DataCodec(S, M)
    L_max = data_codec.L_max
    transport_codec_tmp = TransportCodec(L_max)

    content = app_encode(data, content_type=content_type, filename=filename)
    total_length = len(content)
    K, chunk_size = transport_codec_tmp.get_params(total_length)

    if num_frames is None:
        # 生成足够多的冗余帧（K * 2 + 2，确保 LT peeling decoder 高概率成功）
        num_frames = K * 2 + 2

    images, tc, dc, p, s, g, _, tl = full_encode(
        data, content_type, filename,
        n_colors, n_shapes, level,
        num_frames, module_size, quiet_zone, seed,
    )

    content_out, msg = full_decode(
        images, tc, dc, p, s, g,
        module_size, quiet_zone,
    )

    return content_out, msg, total_length, num_frames


# ======================================================================
# 1. 理想无损全链路往返
# ======================================================================

class TestFullE2EIdeal:
    """理想无损场景：全链路编码 → 渲染 → 采样 → 解码 → 恢复。"""

    @pytest.mark.parametrize("n_colors,n_shapes,level", [
        (4, 4, 1),   # M=24, S=216
        (4, 4, 2),   # M=24, S=616
        (8, 4, 1),   # M=112, S=216
        (2, 4, 2),   # M=2, S=616 (2色档, 等级2)
    ])
    def test_small_file_roundtrip(self, n_colors, n_shapes, level):
        """小文件全链路往返。"""
        data = os.urandom(32)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "test.bin",
            n_colors, n_shapes, level,
        )
        assert content_out is not None, "传输层解码失败"
        assert msg is not None, "应用层解码失败"
        assert msg.data == data
        assert msg.filename == "test.bin"
        assert msg.is_file

    def test_text_message_roundtrip(self):
        """文本消息全链路往返。"""
        text = "Hello, Cimbar! 你好，世界！".encode("utf-8")
        content_out, msg, tl, nf = run_full_e2e(
            text, CONTENT_TYPE_TEXT, "",
            n_colors=4, n_shapes=4, level=1,
        )
        assert msg is not None
        assert msg.is_text
        assert msg.text == "Hello, Cimbar! 你好，世界！"
        assert msg.filename == ""

    def test_empty_file_roundtrip(self):
        """空文件全链路往返。"""
        data = b""
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "empty.dat",
            n_colors=4, n_shapes=4, level=1,
        )
        assert msg is not None
        assert msg.data == b""
        assert msg.filename == "empty.dat"

    def test_large_file_roundtrip(self):
        """较大文件全链路往返（多帧 LT 编码）。"""
        # 使用等级 2 以获得更大 L_max，减少帧数
        data = os.urandom(500)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "large.bin",
            n_colors=4, n_shapes=4, level=2,
        )
        assert msg is not None
        assert msg.data == data

    def test_exact_chunk_size_boundary(self):
        """数据恰好等于 chunk_size 边界。"""
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        sym_enc = SymbolEncoder(pal, ss, grid)
        dc = DataCodec(sym_enc.S, sym_enc.M)
        tc = TransportCodec(dc.L_max)
        chunk_size = tc.chunk_size

        # content 长度恰好 = chunk_size * K（无 padding）
        # app_encode 会加 3 bytes header + filename
        filename = "f.dat"
        raw_data = os.urandom(chunk_size - 3 - len(filename))
        content_out, msg, tl, nf = run_full_e2e(
            raw_data, CONTENT_TYPE_FILE, filename,
            n_colors=4, n_shapes=4, level=1,
        )
        assert msg is not None
        assert msg.data == raw_data


# ======================================================================
# 2. 帧丢失容错（LT 喷泉码）
# ======================================================================

class TestFrameLoss:
    """LT 喷泉码允许丢帧仍能恢复。"""

    def test_with_extra_frames(self):
        """生成多余帧，丢掉一部分仍能解码。"""
        data = os.urandom(200)
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        sym_enc = SymbolEncoder(pal, ss, grid)
        dc = DataCodec(sym_enc.S, sym_enc.M)
        tc = TransportCodec(dc.L_max)
        content = app_encode(data, CONTENT_TYPE_FILE, "test.bin")
        K, _ = tc.get_params(len(content))

        # 生成 3K 帧
        num_frames = K * 3
        images, tc2, dc2, p, s, g, _, _ = full_encode(
            data, CONTENT_TYPE_FILE, "test.bin",
            n_colors=4, n_shapes=4, level=1,
            num_frames=num_frames,
        )

        # 随机丢弃 1/3 的帧（保留 2/3，仍 >= K）
        rng = random.Random(123)
        indices = list(range(len(images)))
        rng.shuffle(indices)
        keep_count = len(images) * 2 // 3
        keep = sorted(indices[:keep_count])
        kept_images = [images[i] for i in keep]

        content_out, msg = full_decode(kept_images, tc2, dc2, p, s, g)
        assert msg is not None
        assert msg.data == data

    def test_minimal_frames(self):
        """恰好 K 帧时也能解码（理想无损）。"""
        data = os.urandom(64)
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        sym_enc = SymbolEncoder(pal, ss, grid)
        dc = DataCodec(sym_enc.S, sym_enc.M)
        tc = TransportCodec(dc.L_max)
        content = app_encode(data, CONTENT_TYPE_FILE, "min.bin")
        K, _ = tc.get_params(len(content))

        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "min.bin",
            n_colors=4, n_shapes=4, level=1,
            num_frames=K,
        )
        assert msg is not None
        assert msg.data == data

    def test_insufficient_frames_fail(self):
        """帧数不足 K 时传输层解码失败。"""
        data = os.urandom(200)
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        sym_enc = SymbolEncoder(pal, ss, grid)
        dc = DataCodec(sym_enc.S, sym_enc.M)
        tc = TransportCodec(dc.L_max)
        content = app_encode(data, CONTENT_TYPE_FILE, "test.bin")
        K, _ = tc.get_params(len(content))

        # 只生成 K-1 帧
        images, tc2, dc2, p, s, g, _, _ = full_encode(
            data, CONTENT_TYPE_FILE, "test.bin",
            n_colors=4, n_shapes=4, level=1,
            num_frames=K - 1,
        )
        content_out, msg = full_decode(images, tc2, dc2, p, s, g)
        assert content_out is None


# ======================================================================
# 3. 噪声退化全链路
# ======================================================================

class TestFullE2ENoise:
    """噪声退化下的全链路：RS + CRC 纠错 + LT 喷泉码容错。"""

    def _e2e_with_noise(self, data, content_type, filename,
                        n_colors, n_shapes, level,
                        noise_intensity=10, module_size=20, seed=42):
        pal = ColorPalette(n_colors)
        ss = ShapeSet(n_shapes)
        grid = Grid(level)
        sym_enc = SymbolEncoder(pal, ss, grid)
        S, M = sym_enc.S, sym_enc.M
        dc = DataCodec(S, M)
        L_max = dc.L_max
        tc = TransportCodec(L_max)
        content = app_encode(data, content_type=content_type, filename=filename)
        K, _ = tc.get_params(len(content))

        # 生成冗余帧
        num_frames = max(K * 2, K + 5)

        images, _, _, p, s, g, _, _ = full_encode(
            data, content_type, filename,
            n_colors, n_shapes, level,
            num_frames, module_size, seed=seed,
        )

        # 添加噪声
        noisy_images = []
        for i, img in enumerate(images):
            noisy = FrameRenderer.add_noise(img, intensity=noise_intensity, seed=seed + i)
            noisy_images.append(noisy)

        content_out, msg = full_decode(noisy_images, tc, dc, p, s, g, module_size)
        return content_out, msg, K, num_frames

    def test_low_noise_recovery(self):
        """低噪声下全链路恢复（RS 纠正单帧错误 + LT 冗余）。"""
        data = os.urandom(64)
        content_out, msg, K, nf = self._e2e_with_noise(
            data, CONTENT_TYPE_FILE, "test.bin",
            n_colors=4, n_shapes=4, level=1,
            noise_intensity=5, module_size=20,
        )
        assert msg is not None, "低噪声下应能恢复"
        assert msg.data == data

    def test_moderate_noise_with_redundancy(self):
        """中等噪声 + 冗余帧仍能恢复。"""
        data = os.urandom(32)
        content_out, msg, K, nf = self._e2e_with_noise(
            data, CONTENT_TYPE_FILE, "test.bin",
            n_colors=4, n_shapes=4, level=1,
            noise_intensity=15, module_size=20,
        )
        # 中等噪声下可能有些帧 RS 失败，但 LT 冗余应能补偿
        assert msg is not None, "中等噪声 + 冗余应能恢复"
        assert msg.data == data


# ======================================================================
# 4. 多 RS 块 / 多 LT 块场景
# ======================================================================

class TestMultiBlock:
    """多 RS 块和多 LT 块的全链路。"""

    def test_multi_rs_block(self):
        """多 RS 块场景（等级 2/3/4 的大 L_max）。"""
        data = os.urandom(300)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "multi_rs.bin",
            n_colors=4, n_shapes=4, level=2,
        )
        assert msg is not None
        assert msg.data == data

    def test_multi_lt_block(self):
        """多 LT 块（content 跨多个 chunk）。"""
        pal = ColorPalette(4)
        ss = ShapeSet(4)
        grid = Grid(1)
        sym_enc = SymbolEncoder(pal, ss, grid)
        dc = DataCodec(sym_enc.S, sym_enc.M)
        tc = TransportCodec(dc.L_max)
        chunk_size = tc.chunk_size

        # content 跨 3 个 chunk
        raw_data = os.urandom(chunk_size * 3 - 10)
        content = app_encode(raw_data, CONTENT_TYPE_FILE, "multi_lt.bin")
        K, _ = tc.get_params(len(content))

        # 小 K 时 peeling decoder 需要更多冗余帧
        num_frames = max(K * 3 + 5, 20)
        content_out, msg, tl, nf = run_full_e2e(
            raw_data, CONTENT_TYPE_FILE, "multi_lt.bin",
            n_colors=4, n_shapes=4, level=1,
            num_frames=num_frames,
        )
        assert msg is not None
        assert msg.data == raw_data

    def test_large_grid_level4(self):
        """等级 4 大网格全链路。"""
        data = os.urandom(256)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "lvl4.bin",
            n_colors=4, n_shapes=4, level=4,
            module_size=8,
        )
        assert msg is not None
        assert msg.data == data

    def test_large_grid_level5(self):
        """等级 5 大网格全链路。"""
        data = os.urandom(256)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "lvl5.bin",
            n_colors=4, n_shapes=4, level=5,
            module_size=8,
        )
        assert msg is not None
        assert msg.data == data

    def test_large_grid_level6(self):
        """等级 6 大网格全链路。"""
        data = os.urandom(256)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "lvl6.bin",
            n_colors=4, n_shapes=4, level=6,
            module_size=8,
        )
        assert msg is not None
        assert msg.data == data

    def test_large_grid_level7(self):
        """等级 7 大网格全链路。"""
        data = os.urandom(256)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "lvl7.bin",
            n_colors=4, n_shapes=4, level=7,
            module_size=8,
        )
        assert msg is not None
        assert msg.data == data

    def test_large_grid_level8(self):
        """等级 8 大网格全链路。"""
        data = os.urandom(256)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "lvl8.bin",
            n_colors=4, n_shapes=4, level=8,
            module_size=8,
        )
        assert msg is not None
        assert msg.data == data

    def test_large_grid_level9(self):
        """等级 9 大网格全链路。"""
        data = os.urandom(256)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "lvl9.bin",
            n_colors=4, n_shapes=4, level=9,
            module_size=8,
        )
        assert msg is not None
        assert msg.data == data

    def test_large_grid_level10(self):
        """等级 10 大网格全链路。"""
        data = os.urandom(256)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "lvl10.bin",
            n_colors=4, n_shapes=4, level=10,
            module_size=8,
        )
        assert msg is not None
        assert msg.data == data

    def test_large_grid_level15(self):
        """等级 15 大网格全链路。"""
        data = os.urandom(256)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "lvl15.bin",
            n_colors=4, n_shapes=4, level=15,
            module_size=8,
        )
        assert msg is not None
        assert msg.data == data


# ======================================================================
# 5. 边界数据
# ======================================================================

class TestBoundaryData:
    """边界数据的全链路。"""

    def test_all_zero_data(self):
        """全零数据。"""
        data = b"\x00" * 64
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "zeros.bin",
            n_colors=4, n_shapes=4, level=1,
        )
        assert msg is not None
        assert msg.data == data

    def test_all_ones_data(self):
        """全 0xFF 数据。"""
        data = b"\xFF" * 64
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "ones.bin",
            n_colors=4, n_shapes=4, level=1,
        )
        assert msg is not None
        assert msg.data == data

    def test_long_filename(self):
        """长文件名。"""
        filename = "a" * 200
        data = os.urandom(32)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, filename,
            n_colors=4, n_shapes=4, level=1,
        )
        assert msg is not None
        assert msg.filename == filename

    def test_unicode_filename(self):
        """Unicode 文件名。"""
        filename = "测试文件_🎉.dat"
        data = os.urandom(16)
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, filename,
            n_colors=4, n_shapes=4, level=1,
        )
        assert msg is not None
        assert msg.filename == filename

    def test_single_byte_data(self):
        """单字节数据。"""
        data = b"\x42"
        content_out, msg, tl, nf = run_full_e2e(
            data, CONTENT_TYPE_FILE, "single.bin",
            n_colors=4, n_shapes=4, level=1,
        )
        assert msg is not None
        assert msg.data == data


# ======================================================================
# 6. 帧顺序无关性
# ======================================================================

class TestFrameOrder:
    """LT 喷泉码不依赖帧顺序。"""

    def test_shuffled_frames(self):
        """打乱帧顺序仍能解码。"""
        data = os.urandom(128)
        images, tc, dc, p, s, g, _, _ = full_encode(
            data, CONTENT_TYPE_FILE, "shuffle.bin",
            n_colors=4, n_shapes=4, level=1,
            num_frames=20,
        )

        rng = random.Random(999)
        rng.shuffle(images)

        content_out, msg = full_decode(images, tc, dc, p, s, g)
        assert msg is not None
        assert msg.data == data

    def test_partial_order(self):
        """部分帧逆序仍能解码。"""
        data = os.urandom(64)
        images, tc, dc, p, s, g, _, _ = full_encode(
            data, CONTENT_TYPE_FILE, "partial.bin",
            n_colors=4, n_shapes=4, level=1,
            num_frames=15,
        )

        # 前半正序，后半逆序
        mid = len(images) // 2
        reordered = images[:mid] + images[mid:][::-1]

        content_out, msg = full_decode(reordered, tc, dc, p, s, g)
        assert msg is not None
        assert msg.data == data
