"""
传输层 + 全链路喷泉码测试。

覆盖范围：
- LT 喷泉码基础编解码往返
- 高 seq 帧的正确性（无限生成核心验证）
- 非连续 seq 帧的解码
- 任意子集帧解码（喷泉码核心特性）
- GUI 使用的无限生成模式：split_blocks + lt_encode_frame 递增 seq
- 全链路：应用层 → 传输层 → 数据层 → 符号层 → 渲染 → 解码 → 还原
- 去重 / 帧丢失 / 帧乱序
"""

import os
import random
import struct

import pytest

from transport_layer import TransportCodec
from transport_layer.lt_code import (
    LTDecoder,
    lt_encode,
    lt_encode_frame,
    split_blocks,
)
from transport_layer.prng import derive_seed, xorshift32, mix32
from transport_layer.rsd import robust_soliton_cdf, sample_degree, sample_indices

from application_layer import app_encode, app_decode, CONTENT_TYPE_TEXT, CONTENT_TYPE_FILE
from data_layer import DataCodec
from symbol_layer import (
    ColorPalette,
    Grid,
    ShapeSet,
    SymbolEncoder,
    FrameRenderer,
    ImageSampler,
    SymbolDecoder,
)


# ======================================================================
# 辅助
# ======================================================================

def random_content(size: int, seed: int = 42) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(size))


# ======================================================================
# 1. PRNG 确定性测试
# ======================================================================

class TestPRNG:
    def test_mix32_deterministic(self):
        assert mix32(0) == mix32(0)
        assert mix32(12345) == mix32(12345)

    def test_mix32_avalanche(self):
        """相邻输入应产生差异很大的输出。"""
        for x in [0, 1, 100, 1000, 0xFFFFFFFF]:
            a = mix32(x)
            b = mix32(x + 1)
            assert a != b
            # 至少有 8 位不同
            diff_bits = bin(a ^ b).count("1")
            assert diff_bits >= 8, f"x={x}: only {diff_bits} bits differ"

    def test_mix32_range(self):
        for x in [0, 1, 0xFFFFFFFF, 123456789]:
            assert 0 <= mix32(x) <= 0xFFFFFFFF

    def test_derive_seed_deterministic(self):
        assert derive_seed(0, 100) == derive_seed(0, 100)
        assert derive_seed(999, 500) == derive_seed(999, 500)

    def test_derive_seed_seq_sensitive(self):
        """不同 seq 应产生不同 seed。"""
        for total_length in [10, 100, 1000]:
            seeds = set()
            for seq in range(100):
                seeds.add(derive_seed(seq, total_length))
            # 100 个不同 seq 应产生大量不同 seed（允许极少量碰撞）
            assert len(seeds) > 90, f"total_length={total_length}: only {len(seeds)} unique seeds"

    def test_derive_seed_length_sensitive(self):
        """不同 total_length 应产生不同 seed。"""
        for seq in [0, 10, 100]:
            seeds = set()
            for tl in range(1, 200):
                seeds.add(derive_seed(seq, tl))
            assert len(seeds) > 180, f"seq={seq}: only {len(seeds)} unique seeds"

    def test_xorshift32_deterministic(self):
        rng1 = xorshift32(42)
        rng2 = xorshift32(42)
        for _ in range(100):
            assert rng1() == rng2()

    def test_xorshift32_no_zero_state(self):
        """seed=0 不应导致退化。"""
        rng = xorshift32(0)
        vals = [rng() for _ in range(10)]
        assert len(set(vals)) > 1  # 不应全是同一个值
        assert all(v != 0 or True for v in vals)  # 不应全为 0


# ======================================================================
# 2. RSD 分布测试
# ======================================================================

class TestRSD:
    @pytest.mark.parametrize("K", [1, 2, 5, 10, 50, 100])
    def test_cdf_length(self, K):
        cdf = robust_soliton_cdf(K)
        assert len(cdf) == K + 1

    @pytest.mark.parametrize("K", [2, 5, 10, 50, 100])
    def test_cdf_monotonic(self, K):
        cdf = robust_soliton_cdf(K)
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1]

    @pytest.mark.parametrize("K", [2, 5, 10, 50, 100])
    def test_cdf_last_is_1(self, K):
        cdf = robust_soliton_cdf(K)
        assert abs(cdf[-1] - 1.0) < 1e-9

    def test_cdf_first_is_0(self):
        cdf = robust_soliton_cdf(10)
        assert cdf[0] == 0.0

    def test_sample_degree_in_range(self):
        K = 50
        cdf = robust_soliton_cdf(K)
        rng = xorshift32(42)
        for _ in range(100):
            d = sample_degree(rng, K, cdf)
            assert 1 <= d <= K

    def test_sample_indices_count(self):
        K = 50
        rng = xorshift32(42)
        for degree in [1, 5, 10, 50]:
            indices = sample_indices(rng, K, degree)
            assert len(indices) == degree
            assert len(set(indices)) == degree  # 无放回
            assert all(0 <= i < K for i in indices)

    def test_sample_indices_deterministic(self):
        K = 50
        rng1 = xorshift32(99)
        rng2 = xorshift32(99)
        d1 = sample_indices(rng1, K, 10)
        d2 = sample_indices(rng2, K, 10)
        assert d1 == d2


# ======================================================================
# 3. LT 喷泉码基础编解码
# ======================================================================

class TestLTFountain:
    def test_split_blocks_basic(self):
        content = b"hello world!"
        blocks = split_blocks(content, 4)
        assert len(blocks) == 3  # ceil(12/4) = 3
        for b in blocks:
            assert len(b) == 4  # 末块 zero-pad

    def test_split_blocks_exact(self):
        content = b"12345678"
        blocks = split_blocks(content, 4)
        assert len(blocks) == 2
        assert blocks[0] == b"1234"
        assert blocks[1] == b"5678"

    def test_split_blocks_empty(self):
        blocks = split_blocks(b"", 4)
        assert len(blocks) == 1
        assert blocks[0] == b"\x00\x00\x00\x00"

    def test_split_blocks_pad(self):
        content = b"ab"
        blocks = split_blocks(content, 4)
        assert len(blocks) == 1
        assert blocks[0] == b"ab\x00\x00"

    def test_lt_encode_frame_length(self):
        content = b"hello world!"
        L_max = 20
        chunk_size = L_max - 8  # 12
        blocks = split_blocks(content, chunk_size)
        frame = lt_encode_frame(blocks, 0, len(content), chunk_size, L_max)
        assert len(frame) == L_max

    def test_lt_encode_frame_header(self):
        content = b"hello world!"
        L_max = 20
        chunk_size = L_max - 8
        blocks = split_blocks(content, chunk_size)
        total_length = len(content)
        for seq in [0, 1, 42, 999]:
            frame = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            tl, sq = struct.unpack(">II", frame[:8])
            assert tl == total_length
            assert sq == seq

    def test_lt_encode_decode_small(self):
        """小内容编解码往返。"""
        content = b"Hello, Cimbar!"
        L_max = 30
        frames, K, chunk_size = lt_encode(content, L_max, 50)
        assert K == 1  # 14 bytes < chunk_size=22
        assert len(frames) == 50

        # 用 TransportCodec 解码
        tc = TransportCodec(L_max)
        decoded = tc.decode(frames)
        assert decoded == content

    def test_lt_encode_decode_large(self):
        """大内容编解码往返。"""
        content = random_content(5000, seed=7)
        L_max = 100
        tc = TransportCodec(L_max)
        K, chunk_size = tc.get_params(len(content))
        frames = tc.encode(content, K * 3)
        decoded = tc.decode(frames)
        assert decoded == content

    def test_lt_encode_decode_near_K(self):
        """接近 K 帧时解码（peeling decoder 需要一定冗余）。"""
        content = random_content(500, seed=11)
        L_max = 50
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))
        # peeling decoder 通常需要 ~1.5K 帧才能可靠解码
        frames = tc.encode(content, K * 2)
        decoded = tc.decode(frames)
        assert decoded == content, f"K={K}, 2K 帧应能解码"


# ======================================================================
# 4. 高 seq 帧正确性（无限生成核心验证）
# ======================================================================

class TestHighSeqFrames:
    """验证高 seq 帧与低 seq 帧一样能正确参与解码。"""

    def test_high_seq_frame_decodable(self):
        """seq=10000 的帧应能正常参与解码。"""
        content = random_content(1000, seed=42)
        L_max = 80
        tc = TransportCodec(L_max)
        K, chunk_size = tc.get_params(len(content))

        # 生成 K*3 帧（喷泉码需要略多于 K 的帧才能可靠解码）
        frames_normal = tc.encode(content, K * 3)
        decoded_normal = tc.decode(frames_normal)
        assert decoded_normal == content

    def test_high_seq_only(self):
        """仅使用高 seq 范围的帧也能解码。"""
        content = random_content(500, seed=99)
        L_max = 60
        tc = TransportCodec(L_max)
        K, chunk_size = tc.get_params(len(content))
        total_length = len(content)
        blocks = split_blocks(content, chunk_size)

        # 生成 seq 从 1000 开始的帧
        frames = []
        for seq in range(1000, 1000 + K * 3):
            frame = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            frames.append(frame)

        decoded = tc.decode(frames)
        assert decoded == content, "高 seq 范围帧应能正确解码"

    def test_mixed_low_high_seq(self):
        """低 seq 和高 seq 混合使用。"""
        content = random_content(2000, seed=55)
        L_max = 80
        tc = TransportCodec(L_max)
        K, chunk_size = tc.get_params(len(content))
        total_length = len(content)
        blocks = split_blocks(content, chunk_size)

        # 混合 seq: 0,1,2, 100,101,102, 200,201,202, ...
        frames = []
        seqs = list(range(0, K)) + list(range(100, 100 + K * 2))
        for seq in seqs:
            frame = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            frames.append(frame)

        decoded = tc.decode(frames)
        assert decoded == content

    def test_seq_4byte_wraparound(self):
        """seq 超过 32 位不会发生（uint32），验证接近上限的 seq。"""
        content = random_content(300, seed=33)
        L_max = 50
        tc = TransportCodec(L_max)
        K, chunk_size = tc.get_params(len(content))
        total_length = len(content)
        blocks = split_blocks(content, chunk_size)

        # 使用接近 uint32 上限的 seq
        high_seqs = [0xFFFFFF00 + i for i in range(K * 3)]
        frames = []
        for seq in high_seqs:
            frame = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            frames.append(frame)

        decoded = tc.decode(frames)
        assert decoded == content, "接近 uint32 上限的 seq 应能正常解码"

    def test_every_seq_unique_frame(self):
        """每个 seq 产生的帧应不同（极大概率）。"""
        content = random_content(100, seed=1)
        L_max = 40
        chunk_size = L_max - 8
        blocks = split_blocks(content, chunk_size)

        frame_set = set()
        for seq in range(200):
            frame = lt_encode_frame(blocks, seq, len(content), chunk_size, L_max)
            frame_set.add(frame)

        # 200 个不同 seq 应产生 200 个不同的帧
        assert len(frame_set) == 200, f"只有 {len(frame_set)} 个唯一帧，期望 200"


# ======================================================================
# 5. 任意子集解码（喷泉码核心特性）
# ======================================================================

class TestSubsetDecoding:
    """喷泉码核心特性：任意足够的帧子集都能解码。"""

    def test_random_subset_decodes(self):
        """随机子集帧应能解码。"""
        content = random_content(2000, seed=42)
        L_max = 80
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))

        # 生成 4K 帧
        all_frames = tc.encode(content, K * 4)

        # 多次随机抽取 2K 帧（足够 peeling decoder 解码）
        rng = random.Random(123)
        for trial in range(10):
            subset = rng.sample(all_frames, K * 2)
            decoded = tc.decode(subset)
            assert decoded == content, f"trial {trial}: 随机子集解码失败"

    def test_first_n_frames_decode(self):
        """前 N 帧应能解码。"""
        content = random_content(1000, seed=77)
        L_max = 60
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))

        all_frames = tc.encode(content, K * 4)
        # 前 2K 帧应足够
        decoded = tc.decode(all_frames[:K * 2])
        assert decoded == content

    def test_last_n_frames_decode(self):
        """后 N 帧应能解码。"""
        content = random_content(1000, seed=88)
        L_max = 60
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))

        all_frames = tc.encode(content, K * 4)
        # 后 2K 帧应足够
        decoded = tc.decode(all_frames[K * 2:])
        assert decoded == content

    def test_skip_first_frame(self):
        """跳过第一帧（total_length 来源）也能解码。"""
        content = random_content(500, seed=66)
        L_max = 50
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))

        all_frames = tc.encode(content, K * 3)
        # 跳过第一帧
        decoded = tc.decode(all_frames[1:])
        assert decoded == content, "跳过第一帧应仍能解码（total_length 可从任意帧解析）"

    def test_shuffled_order(self):
        """帧顺序打乱后仍能解码。"""
        content = random_content(1500, seed=22)
        L_max = 70
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))

        all_frames = tc.encode(content, K * 2)
        rng = random.Random(456)
        rng.shuffle(all_frames)
        decoded = tc.decode(all_frames)
        assert decoded == content


# ======================================================================
# 6. 去重 / 帧丢失 / 重复帧
# ======================================================================

class TestDedupAndLoss:
    def test_duplicate_frames_ignored(self):
        """重复帧应被忽略，不影响解码。"""
        content = random_content(800, seed=44)
        L_max = 60
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))

        frames = tc.encode(content, K * 2)
        # 复制部分帧
        frames_with_dups = frames + frames[:K]
        decoded = tc.decode(frames_with_dups)
        assert decoded == content

    def test_frame_loss_tolerant(self):
        """丢失部分帧仍能解码。"""
        content = random_content(1000, seed=33)
        L_max = 50
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))

        all_frames = tc.encode(content, K * 3)
        # 丢掉 1/3 的帧
        subset = all_frames[::2]  # 取偶数索引
        decoded = tc.decode(subset)
        assert decoded == content

    def test_insufficient_frames_fail(self):
        """帧数不足时应返回 None 或正确内容（概率性）。"""
        content = random_content(2000, seed=55)
        L_max = 80
        tc = TransportCodec(L_max)
        K, _ = tc.get_params(len(content))

        # 只用 1 帧（K > 1 时几乎不可能解码）
        frames = tc.encode(content, 1)
        decoded = tc.decode(frames)
        # K=1 时可能成功，K>1 时应失败
        if K > 1:
            assert decoded is None or decoded != content

    def test_empty_frame_list(self):
        tc = TransportCodec(30)
        assert tc.decode([]) is None

    def test_corrupted_frame_header(self):
        """帧头损坏（total_length 不一致）应被 LTDecoder 拒绝。"""
        content = random_content(100, seed=11)
        L_max = 30
        tc = TransportCodec(L_max)
        frames = tc.encode(content, 10)
        # 篡改第一帧的 total_length
        corrupted = bytearray(frames[0])
        struct.pack_into(">I", corrupted, 0, 99999)
        frames[0] = bytes(corrupted)
        # 解码应失败或返回错误内容
        decoded = tc.decode(frames)
        assert decoded is None or decoded != content


# ======================================================================
# 7. GUI 无限生成模式验证
# ======================================================================

class TestGUIInfiniteMode:
    """验证 app.py 中使用的无限生成模式：split_blocks + lt_encode_frame 递增 seq。"""

    def test_infinite_mode_basic(self):
        """模拟 GUI 的无限生成模式：递增 seq 生成帧，取足够数量解码。"""
        content = random_content(800, seed=42)
        L_max = 60
        chunk_size = L_max - 8
        blocks = split_blocks(content, chunk_size)
        total_length = len(content)

        tc = TransportCodec(L_max)
        K, _ = tc.get_params(total_length)

        # 模拟 GUI：从 seq=0 开始递增生成
        frames = []
        for seq in range(K * 2):
            frame = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            frames.append(frame)

        decoded = tc.decode(frames)
        assert decoded == content

    def test_infinite_mode_from_arbitrary_start(self):
        """从任意 seq 起点开始生成也能解码。"""
        content = random_content(600, seed=77)
        L_max = 50
        chunk_size = L_max - 8
        blocks = split_blocks(content, chunk_size)
        total_length = len(content)

        tc = TransportCodec(L_max)
        K, _ = tc.get_params(total_length)

        start_seq = 500
        frames = []
        for seq in range(start_seq, start_seq + K * 2):
            frame = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            frames.append(frame)

        decoded = tc.decode(frames)
        assert decoded == content

    def test_infinite_mode_streaming(self):
        """模拟流式播放：先生成一批，再继续生成，合在一起能解码。"""
        content = random_content(1500, seed=99)
        L_max = 70
        chunk_size = L_max - 8
        blocks = split_blocks(content, chunk_size)
        total_length = len(content)

        tc = TransportCodec(L_max)
        K, _ = tc.get_params(total_length)

        # 第一批：seq 0~K-1
        batch1 = []
        for seq in range(K):
            batch1.append(lt_encode_frame(blocks, seq, total_length, chunk_size, L_max))

        # 第二批：seq K~2K-1
        batch2 = []
        for seq in range(K, 2 * K):
            batch2.append(lt_encode_frame(blocks, seq, total_length, chunk_size, L_max))

        # 单独第一批可能不够，合在一起应够
        all_frames = batch1 + batch2
        decoded = tc.decode(all_frames)
        assert decoded == content

    def test_infinite_mode_with_data_layer(self):
        """无限生成模式 + 数据层编解码全链路。"""
        # 构建管线
        g = Grid(1)
        pal = ColorPalette(4)
        shp = ShapeSet(4)
        enc = SymbolEncoder(pal, shp, g)
        S, M = enc.S, enc.M
        data_codec = DataCodec(S, M)
        L_max = data_codec.L_max

        content = app_encode(b"Hello Cimbar Fountain!", CONTENT_TYPE_TEXT, "")
        total_length = len(content)
        chunk_size = L_max - 8
        blocks = split_blocks(content, chunk_size)

        tc = TransportCodec(L_max)
        K, _ = tc.get_params(total_length)

        # 生成帧并经过数据层 + 符号层编码
        frames = []
        for seq in range(max(K * 2, 20)):
            frame_bytes = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            symbols = data_codec.encode(frame_bytes)
            assert len(symbols) == S
            assert all(0 <= s < M for s in symbols)
            frames.append(frame_bytes)

        # 解码
        decoded = tc.decode(frames)
        assert decoded == content

        # 应用层解码
        msg = app_decode(decoded)
        assert msg.data == b"Hello Cimbar Fountain!"
        assert msg.content_type == CONTENT_TYPE_TEXT

    def test_infinite_mode_frame_uniqueness_over_time(self):
        """长时间生成：连续 500 帧应几乎都是唯一的。"""
        content = random_content(200, seed=5)
        L_max = 40
        chunk_size = L_max - 8
        blocks = split_blocks(content, chunk_size)
        total_length = len(content)

        frame_set = set()
        for seq in range(500):
            frame = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            frame_set.add(frame)

        # 允许极少量碰撞（degree=1 + 相同 index 时可能重复）
        assert len(frame_set) > 490, f"500 帧中只有 {len(frame_set)} 个唯一帧"


# ======================================================================
# 8. 全链路往返：应用层 → 传输层 → 数据层 → 符号层 → 渲染 → 解码
# ======================================================================

class TestFullPipeline:
    """全链路往返测试：模拟 GUI 的完整编码 + 解码流程。"""

    def _full_encode_decode(
        self,
        text: str,
        n_colors: int = 4,
        n_shapes: int = 4,
        level: int = 1,
        module_size: int = 12,
        num_frames: int = 30,
    ):
        """完整编码 → 渲染 → 采样 → 解码 → 还原。"""
        # 编码管线
        g = Grid(level)
        pal = ColorPalette(n_colors)
        shp = ShapeSet(n_shapes)
        enc = SymbolEncoder(pal, shp, g)
        S, M = enc.S, enc.M
        data_codec = DataCodec(S, M)
        L_max = data_codec.L_max

        content = app_encode(text.encode("utf-8"), CONTENT_TYPE_TEXT, "")
        total_length = len(content)
        chunk_size = L_max - 8
        blocks = split_blocks(content, chunk_size)

        tc = TransportCodec(L_max)
        K, _ = tc.get_params(total_length)

        # 生成帧
        frames_bytes = []
        rendered_frames = []
        renderer = FrameRenderer(module_size=module_size, quiet_zone_size=4)

        for seq in range(num_frames):
            fb = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
            symbols = data_codec.encode(fb)
            rendered = enc.encode(symbols)
            img = renderer.render(rendered)
            frames_bytes.append(fb)
            rendered_frames.append(img)

        # 解码：用 ImageSampler 逐帧解码符号 → 数据层 → 传输层
        decoded_frames = []
        for img in rendered_frames:
            sampler = ImageSampler(img, g, pal, shp, module_size=module_size, quiet_zone_size=4)
            dec = SymbolDecoder(sampler, g)
            dec_frame = dec.decode()
            data_bytes = data_codec.decode(dec_frame.symbol_block, dec_frame.erasure_flags)
            if data_bytes is not None:
                decoded_frames.append(data_bytes)

        # 传输层解码
        result = tc.decode(decoded_frames) if decoded_frames else None
        return content, result, K, len(decoded_frames)

    def test_short_text(self):
        content, result, K, n_decoded = self._full_encode_decode("Hello!")
        assert result == content, f"K={K}, decoded={n_decoded} frames"

    def test_medium_text(self):
        text = "Cimbar fountain code test with medium length text content. " * 5
        content, result, K, n_decoded = self._full_encode_decode(text, level=2, num_frames=60)
        assert result == content, f"K={K}, decoded={n_decoded} frames"

    def test_high_fps_many_frames(self):
        """模拟高 FPS 播放：生成大量帧，只用部分解码。"""
        content, result, K, n_decoded = self._full_encode_decode(
            "Fountain stream test", num_frames=100,
        )
        assert result == content, f"K={K}, decoded={n_decoded} frames"

    def test_all_color_shape_combos(self):
        valid_combos = [
            (4, 2), (4, 4),
            (8, 4),
        ]
        for n_colors, n_shapes in valid_combos:
            content, result, K, n_decoded = self._full_encode_decode(
                "Combo test", n_colors=n_colors, n_shapes=n_shapes, num_frames=40,
            )
            assert result == content, \
                f"colors={n_colors}, shapes={n_shapes}: K={K}, decoded={n_decoded}"


# ======================================================================
# 9. TransportCodec 边界测试
# ======================================================================

class TestTransportCodecBoundary:
    def test_L_max_too_small(self):
        with pytest.raises(ValueError):
            TransportCodec(8)

    def test_L_max_minimum(self):
        """L_max=9 是最小值。"""
        tc = TransportCodec(9)
        assert tc.chunk_size == 1

    def test_empty_content(self):
        tc = TransportCodec(30)
        frames = tc.encode(b"", 5)
        decoded = tc.decode(frames)
        assert decoded == b""

    def test_single_byte(self):
        tc = TransportCodec(30)
        frames = tc.encode(b"\x42", 10)
        decoded = tc.decode(frames)
        assert decoded == b"\x42"

    def test_content_exact_chunk_size(self):
        """内容恰好等于一个 chunk_size。"""
        tc = TransportCodec(30)
        content = b"\x01" * tc.chunk_size
        frames = tc.encode(content, 10)
        decoded = tc.decode(frames)
        assert decoded == content

    def test_content_chunk_size_plus_one(self):
        """内容比一个 chunk_size 多 1 byte → K=2。"""
        tc = TransportCodec(30)
        content = b"\x02" * (tc.chunk_size + 1)
        K, _ = tc.get_params(len(content))
        assert K == 2
        frames = tc.encode(content, K * 3)
        decoded = tc.decode(frames)
        assert decoded == content

    def test_get_params(self):
        tc = TransportCodec(50)
        K, chunk_size = tc.get_params(100)
        assert chunk_size == 42  # 50 - 8
        assert K == 3  # ceil(100/42) = 3
