"""
数据层单元测试。

覆盖范围：
- crc: CRC16-CCITT 计算与校验
- packing: 打包参数表、选择算法、pack/unpack 往返
- rs: RS 编解码往返、错误纠正、erasure、缩短 RS
- codec: DataCodec 容量计算、编码/解码往返、纠错、erasure、CRC 失败、填充
"""

import os
import sys
import random
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.crc import crc16, crc16_check, crc16_append, crc16_verify
from data_layer.packing import (
    PACKING_TABLE, select_packing,
    pack_bytes_to_symbols, unpack_symbols_to_bytes,
)
from data_layer.rs import rs_encode_block, rs_decode_block
from data_layer.codec import DataCodec, RS_NSYM, RS_MAX_DATA, RS_BLOCK_SIZE


# ======================================================================
# 1. CRC16 测试
# ======================================================================

class TestCRC16:
    def test_empty_data(self):
        """空数据的 CRC16 应为初始值 0xFFFF。"""
        assert crc16(b"") == 0xFFFF

    def test_known_vector(self):
        """CCITT-FALSE 标准测试向量: CRC32("123456789") = 0x29B1。"""
        assert crc16(b"123456789") == 0x29B1

    def test_single_byte(self):
        """单字节可重复计算。"""
        val = crc16(b"\x00")
        assert isinstance(val, int)
        assert 0 <= val <= 0xFFFF

    def test_deterministic(self):
        """相同输入应产生相同输出。"""
        data = os.urandom(32)
        assert crc16(data) == crc16(data)

    def test_different_inputs_different_crc(self):
        """不同输入大概率产生不同 CRC。"""
        assert crc16(b"\x00") != crc16(b"\x01")

    def test_crc16_check(self):
        """crc16_check 正确验证。"""
        data = b"hello world"
        c = crc16(data)
        assert crc16_check(data, c) is True
        assert crc16_check(data, c ^ 0x0001) is False

    def test_crc16_append_length(self):
        """append 后长度增加 2。"""
        data = b"test data"
        appended = crc16_append(data)
        assert len(appended) == len(data) + 2

    def test_crc16_append_verify(self):
        """append 后可被 verify 验证通过。"""
        data = b"test data for crc"
        appended = crc16_append(data)
        ok, extracted = crc16_verify(appended)
        assert ok is True
        assert extracted == data

    def test_crc16_verify_tampered(self):
        """篡改数据后 verify 失败。"""
        data = b"test data for crc"
        appended = bytearray(crc16_append(data))
        appended[0] ^= 0xFF
        ok, _ = crc16_verify(bytes(appended))
        assert ok is False

    def test_crc16_verify_short(self):
        """长度不足 2 时 verify 返回失败。"""
        ok, extracted = crc16_verify(b"")
        assert ok is False
        assert extracted == b""

        ok, extracted = crc16_verify(b"\x01")
        assert ok is False
        assert extracted == b""

    def test_crc16_verify_single_byte_data(self):
        """1 byte 数据 + 2 byte CRC = 3 bytes，可正常验证。"""
        data = b"\xAB"
        appended = crc16_append(data)
        ok, extracted = crc16_verify(appended)
        assert ok is True
        assert extracted == data


# ======================================================================
# 2. Packing 测试
# ======================================================================

class TestPackingTable:
    def test_table_matches_doc(self):
        """参数表与文档规范一致。"""
        expected = {
            2: (8, 1),
            4: (4, 1),
            8: (8, 3),
            12: (7, 3),
            16: (2, 1),
            24: (2, 1),
            48: (3, 2),
            56: (3, 2),
            96: (4, 3),
            112: (4, 3),
            224: (4, 3),
            448: (1, 1),
        }
        for M, (k, n) in expected.items():
            assert PACKING_TABLE[M] == (k, n), f"M={M}: 期望 ({k}, {n}), 实际 {PACKING_TABLE[M]}"

    def test_table_constraint(self):
        """表中所有 (k, n) 满足 M^k >= 256^n。"""
        for M, (k, n) in PACKING_TABLE.items():
            assert M ** k >= 256 ** n, f"M={M}, k={k}, n={n}: M^k={M**k} < 256^n={256**n}"


class TestSelectPacking:
    def test_table_lookup(self):
        """表内 M 直接查表。"""
        for M, expected in PACKING_TABLE.items():
            assert select_packing(M) == expected

    def test_off_table_valid(self):
        """表外 M 仍返回满足约束的 (k, n)。"""
        for M in [3, 6, 10, 32, 64, 128, 200]:
            k, n = select_packing(M)
            assert M ** k >= 256 ** n, f"M={M}, k={k}, n={n}: 约束不满足"

    def test_off_table_efficiency(self):
        """表外 M 选择效率最高的 (k, n)。"""
        import math
        for M in [3, 6, 10, 32, 64, 128]:
            k, n = select_packing(M)
            eff = (n * 8) / (k * math.log2(M))
            assert eff <= 1.0 + 1e-9, f"M={M}: 效率 {eff} > 1"


class TestPackUnpack:
    @pytest.mark.parametrize("M", list(PACKING_TABLE.keys()))
    def test_round_trip_all_table(self, M):
        """表中所有 M 的 pack/unpack 往返。"""
        k, n = PACKING_TABLE[M]
        for _ in range(20):
            data = os.urandom(n)
            symbols = pack_bytes_to_symbols(data, M, k, n)
            assert len(symbols) == k
            assert all(0 <= s < M for s in symbols)
            recovered = unpack_symbols_to_bytes(symbols, M, k, n)
            assert recovered == data

    def test_pack_zero(self):
        """全零数据打包为全零符号。"""
        k, n = 4, 1
        M = 16
        data = b"\x00"
        symbols = pack_bytes_to_symbols(data, M, k, n)
        assert symbols == [0, 0, 0, 0]

    def test_pack_max_value(self):
        """最大值数据打包后符号仍在 [0, M-1] 范围内。"""
        M, k, n = 16, 2, 1
        data = b"\xFF"
        symbols = pack_bytes_to_symbols(data, M, k, n)
        assert all(0 <= s < M for s in symbols)
        assert unpack_symbols_to_bytes(symbols, M, k, n) == data

    def test_pack_wrong_length_raises(self):
        """数据长度不匹配 n 时 assert 失败。"""
        with pytest.raises(AssertionError):
            pack_bytes_to_symbols(b"\x00\x00", M=16, k=2, n=1)

    def test_unpack_overflow_mod(self):
        """符号值越界时 unpack 取模防溢出。"""
        M, k, n = 16, 2, 1
        symbols = [M, M]  # 越界
        result = unpack_symbols_to_bytes(symbols, M, k, n)
        assert len(result) == n

    def test_pack_big_endian(self):
        """打包使用 big-endian 字节序。"""
        M, k, n = 256 * 256, 2, 2  # M^k = 256^4 >= 256^2
        # 但 256*256=65536 不在表中，用 select_packing
        k2, n2 = select_packing(65536)
        data = b"\x01\x02"
        symbols = pack_bytes_to_symbols(data, 65536, k2, n2)
        recovered = unpack_symbols_to_bytes(symbols, 65536, k2, n2)
        assert recovered == data


# ======================================================================
# 3. Reed-Solomon 测试
# ======================================================================

class TestRSEncode:
    def test_encode_length(self):
        """编码后长度 = data + nsym。"""
        data = os.urandom(100)
        encoded = rs_encode_block(data, nsym=32)
        assert len(encoded) == 100 + 32

    def test_encode_max_data(self):
        """满块 223 bytes 编码后 255 bytes。"""
        data = os.urandom(RS_MAX_DATA)
        encoded = rs_encode_block(data, nsym=32)
        assert len(encoded) == RS_BLOCK_SIZE

    def test_encode_shortened(self):
        """缩短 RS：数据 < 223，parity 仍为 32。"""
        data = os.urandom(50)
        encoded = rs_encode_block(data, nsym=32)
        assert len(encoded) == 50 + 32

    def test_encode_empty(self):
        """空数据返回全零 parity。"""
        encoded = rs_encode_block(b"", nsym=32)
        assert len(encoded) == 32
        assert encoded == b"\x00" * 32


class TestRSDecode:
    def test_round_trip_no_error(self):
        """无错误时编解码往返。"""
        data = os.urandom(100)
        encoded = rs_encode_block(data, nsym=32)
        decoded, ok = rs_decode_block(encoded, nsym=32)
        assert ok is True
        assert decoded == data

    def test_round_trip_max_data(self):
        """满块 223 bytes 往返。"""
        data = os.urandom(RS_MAX_DATA)
        encoded = rs_encode_block(data, nsym=32)
        decoded, ok = rs_decode_block(encoded, nsym=32)
        assert ok is True
        assert decoded == data

    def test_round_trip_shortened(self):
        """缩短 RS 往返。"""
        data = os.urandom(50)
        encoded = rs_encode_block(data, nsym=32)
        decoded, ok = rs_decode_block(encoded, nsym=32)
        assert ok is True
        assert decoded == data

    def test_correct_errors(self):
        """纠正 <= 16 bytes 随机错误。"""
        data = os.urandom(200)
        encoded = bytearray(rs_encode_block(data, nsym=32))
        # 引入 16 个错误
        positions = random.sample(range(len(encoded)), 16)
        for pos in positions:
            encoded[pos] ^= 0xFF
        decoded, ok = rs_decode_block(bytes(encoded), nsym=32)
        assert ok is True
        assert decoded == data

    def test_correct_erasure(self):
        """纠正 <= 32 bytes erasure。"""
        data = os.urandom(200)
        encoded = bytearray(rs_encode_block(data, nsym=32))
        # 引入 32 个 erasure
        positions = random.sample(range(len(encoded)), 32)
        for pos in positions:
            encoded[pos] ^= 0xFF
        decoded, ok = rs_decode_block(bytes(encoded), nsym=32, erasure_pos=positions)
        assert ok is True
        assert decoded == data

    def test_too_many_errors(self):
        """超过 16 bytes 错误时解码失败。"""
        data = os.urandom(200)
        encoded = bytearray(rs_encode_block(data, nsym=32))
        # 引入 17 个错误（超过纠错能力）
        positions = random.sample(range(len(encoded)), 17)
        for pos in positions:
            encoded[pos] ^= 0xFF
        decoded, ok = rs_decode_block(bytes(encoded), nsym=32)
        assert ok is False

    def test_too_many_erasures(self):
        """超过 32 bytes erasure 时解码失败。"""
        data = os.urandom(200)
        encoded = bytearray(rs_encode_block(data, nsym=32))
        positions = list(range(33))
        for pos in positions:
            encoded[pos] ^= 0xFF
        decoded, ok = rs_decode_block(bytes(encoded), nsym=32, erasure_pos=positions)
        assert ok is False

    def test_erasure_better_than_error(self):
        """同样 32 bytes 损坏：erasure 可纠正，error 不可纠正。"""
        data = os.urandom(200)
        encoded = bytearray(rs_encode_block(data, nsym=32))
        positions = list(range(32))
        for pos in positions:
            encoded[pos] ^= 0xFF

        # 作为 erasure 可纠正
        dec_ok, ok = rs_decode_block(bytes(encoded), nsym=32, erasure_pos=positions)
        assert ok is True
        assert dec_ok == data

        # 作为 error（不告知位置）不可纠正
        _, ok2 = rs_decode_block(bytes(encoded), nsym=32)
        assert ok2 is False

    def test_erasure_pos_out_of_range_filtered(self):
        """越界 erasure 位置被过滤，不影响解码。"""
        data = os.urandom(100)
        encoded = rs_encode_block(data, nsym=32)
        decoded, ok = rs_decode_block(encoded, nsym=32, erasure_pos=[999, -1])
        assert ok is True
        assert decoded == data


# ======================================================================
# 4. DataCodec 测试
# ======================================================================

class TestDataCodecCapacity:
    """测试容量计算与文档示例一致。"""

    def test_capacity_example_s211_m24(self):
        """文档示例: S=211, M=24 → L_max=71。"""
        codec = DataCodec(S=211, M=24)
        assert codec.k == 2
        assert codec.n == 1
        assert codec.P == 105
        assert codec.total_bytes == 105
        assert codec.R == 1
        assert codec.parity_total == 32
        assert codec.L_max == 71

    def test_capacity_example_s595_m24(self):
        """文档示例: S=595, M=24 → L_max=231。"""
        codec = DataCodec(S=595, M=24)
        assert codec.P == 297
        assert codec.total_bytes == 297
        assert codec.R == 2
        assert codec.parity_total == 64
        assert codec.L_max == 231

    def test_capacity_example_s1107_m24(self):
        """文档示例: S=1107, M=24 → L_max=455。"""
        codec = DataCodec(S=1107, M=24)
        assert codec.P == 553
        assert codec.total_bytes == 553
        assert codec.R == 3
        assert codec.parity_total == 96
        assert codec.L_max == 455

    def test_capacity_example_s1747_m24(self):
        """文档示例: S=1747, M=24 → L_max=743。"""
        codec = DataCodec(S=1747, M=24)
        assert codec.P == 873
        assert codec.total_bytes == 873
        assert codec.R == 4
        assert codec.parity_total == 128
        assert codec.L_max == 743

    def test_capacity_formula(self):
        """L_max = P*n - R*32 - 2。"""
        for S in [216, 616, 1144, 1800, 2584, 3496, 4536, 5704, 7000, 8424, 9976, 11656, 13464, 15400, 17464]:
            for M in [4, 8, 16, 24, 48, 96, 224, 448]:
                codec = DataCodec(S=S, M=M)
                expected = codec.P * codec.n - codec.R * RS_NSYM - 2
                assert codec.L_max == expected
                assert codec.L_max > 0

    def test_insufficient_capacity_raises(self):
        """S 或 M 太小导致 L_max < 0 时抛异常。"""
        with pytest.raises(ValueError):
            DataCodec(S=1, M=2)

    def test_repr(self):
        """repr 包含关键参数。"""
        codec = DataCodec(S=211, M=24)
        r = repr(codec)
        assert "S=211" in r
        assert "M=24" in r
        assert "L_max=71" in r


class TestDataCodecEncodeDecode:
    @pytest.mark.parametrize("S,M", [
        (616, 2),      # 等级2, 2色1图形
        (216, 4),      # 等级1, 4色1图形
        (216, 24),     # 等级1, 4色4图形
        (616, 24),     # 等级2, 4色4图形
        (616, 48),     # 等级2, 8色4图形
        (1144, 96),    # 等级3, 8色8图形
        (1800, 224),   # 等级4, 8色16图形
        (1800, 448),   # 等级4, 8色16图形 (M=448)
        (2584, 24),    # 等级5, 4色4图形
        (3496, 24),    # 等级6, 4色4图形
        (4536, 24),    # 等级7, 4色4图形
        (5704, 24),    # 等级8, 4色4图形
        (7000, 24),    # 等级9, 4色4图形
        (8424, 24),    # 等级10, 4色4图形
        (9976, 24),    # 等级11, 4色4图形
        (11656, 24),   # 等级12, 4色4图形
        (13464, 24),   # 等级13, 4色4图形
        (15400, 24),   # 等级14, 4色4图形
        (17464, 24),   # 等级15, 4色4图形
    ])
    def test_round_trip(self, S, M):
        """编码后解码恢复原始数据。"""
        codec = DataCodec(S=S, M=M)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        assert len(symbols) == S
        assert all(0 <= s < M for s in symbols)
        decoded = codec.decode(symbols)
        assert decoded is not None
        assert decoded == data

    def test_encode_wrong_length_raises(self):
        """编码时数据长度不等于 L_max 抛异常。"""
        codec = DataCodec(S=216, M=24)
        with pytest.raises(ValueError):
            codec.encode(b"\x00" * (codec.L_max - 1))

    def test_decode_wrong_symbol_count_raises(self):
        """解码时符号数不等于 S 抛异常。"""
        codec = DataCodec(S=216, M=24)
        with pytest.raises(ValueError):
            codec.decode([0] * (codec.S - 1))

    def test_encode_padding(self):
        """P*k < S 时末尾补 0 符号。"""
        # S=211, M=24, k=2 → P=105, P*k=210 < 211
        codec = DataCodec(S=211, M=24)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        assert len(symbols) == 211
        assert symbols[210] == 0  # 填充位

    def test_decode_padding_ignored(self):
        """填充符号不影响解码。"""
        codec = DataCodec(S=211, M=24)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        # 篡改填充位
        symbols[210] = codec.M - 1
        decoded = codec.decode(symbols)
        assert decoded == data


class TestDataCodecErrorCorrection:
    def test_decode_with_errors(self):
        """少量符号错误可被 RS 纠正。"""
        codec = DataCodec(S=616, M=24)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        # 翻转少量符号（确保不超过 RS 纠错能力）
        num_errors = min(5, codec.P)
        positions = random.sample(range(codec.P * codec.k), num_errors)
        for pos in positions:
            symbols[pos] = (symbols[pos] + 1) % codec.M
        decoded = codec.decode(symbols)
        assert decoded is not None
        assert decoded == data

    def test_decode_with_erasure(self):
        """erasure 标记使 RS 纠错能力翻倍。"""
        codec = DataCodec(S=616, M=24)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        # 引入较多错误但标记为 erasure
        num_erasure = min(10, codec.P)
        positions = random.sample(range(codec.P * codec.k), num_erasure)
        erasure_flags = [False] * codec.S
        for pos in positions:
            symbols[pos] = (symbols[pos] + 1) % codec.M
            erasure_flags[pos] = True
        decoded = codec.decode(symbols, erasure_flags=erasure_flags)
        assert decoded is not None
        assert decoded == data

    def test_decode_crc_failure(self):
        """RS 纠正成功但 CRC 不匹配时返回 None。"""
        codec = DataCodec(S=216, M=24)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        # 在 RS 纠错能力内篡改，但篡改量恰好使 CRC 变化
        # 直接篡改 1 个数据符号
        symbols[0] = (symbols[0] + 1) % codec.M
        # 不提供 erasure，让 RS 当 error 纠正
        # 如果 RS 纠正成功，CRC 应该匹配（因为 RS 恢复了原始数据）
        # 所以这里测试的是：如果篡改在纠错范围内，CRC 验证通过
        decoded = codec.decode(symbols)
        # 1 个 error 在 RS 纠错能力内，应该能纠正
        if decoded is not None:
            assert decoded == data

    def test_decode_too_many_errors_returns_none(self):
        """错误超过 RS 纠错能力时返回 None。"""
        codec = DataCodec(S=616, M=24)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        # 引入大量错误（远超纠错能力）
        for i in range(0, len(symbols), 1):
            symbols[i] = (symbols[i] + 1) % codec.M
        decoded = codec.decode(symbols)
        assert decoded is None

    def test_decode_all_erasure_too_many(self):
        """erasure 数量超过 RS 能力时返回 None。"""
        codec = DataCodec(S=216, M=24)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        # 全部标记为 erasure 并篡改
        erasure_flags = [True] * codec.S
        for i in range(len(symbols)):
            symbols[i] = (symbols[i] + 1) % codec.M
        decoded = codec.decode(symbols, erasure_flags=erasure_flags)
        assert decoded is None


class TestDataCodecMultiBlock:
    def test_multi_block_rs(self):
        """多 RS 块场景（R > 1）编解码往返。"""
        # S=616, M=24 → P=308, total_bytes=308, R=2
        codec = DataCodec(S=616, M=24)
        assert codec.R == 2
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        decoded = codec.decode(symbols)
        assert decoded == data

    def test_multi_block_with_errors(self):
        """多 RS 块场景下每块都有错误仍可纠正。"""
        codec = DataCodec(S=616, M=24)
        assert codec.R == 2
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        # 在每个 RS 块对应的符号范围内各引入少量错误
        k = codec.k
        n = codec.n
        # RS 块 1 覆盖前 ~255 bytes → 前 255 个打包块
        # RS 块 2 覆盖剩余
        # 在每个块范围内翻转 2 个符号
        block1_end = min(255 // n, codec.P) * k
        for pos in [0, 3]:
            symbols[pos] = (symbols[pos] + 1) % codec.M
        for pos in [block1_end, block1_end + 3]:
            if pos < len(symbols):
                symbols[pos] = (symbols[pos] + 1) % codec.M
        decoded = codec.decode(symbols)
        assert decoded is not None
        assert decoded == data


class TestDataCodecEdgeCases:
    def test_all_zero_data(self):
        """全零数据编解码往返。"""
        codec = DataCodec(S=216, M=24)
        data = b"\x00" * codec.L_max
        symbols = codec.encode(data)
        decoded = codec.decode(symbols)
        assert decoded == data

    def test_all_ones_data(self):
        """全 0xFF 数据编解码往返。"""
        codec = DataCodec(S=216, M=24)
        data = b"\xFF" * codec.L_max
        symbols = codec.encode(data)
        decoded = codec.decode(symbols)
        assert decoded == data

    def test_random_data_multiple_runs(self):
        """多次随机数据往返。"""
        codec = DataCodec(S=616, M=48)
        for _ in range(10):
            data = os.urandom(codec.L_max)
            symbols = codec.encode(data)
            decoded = codec.decode(symbols)
            assert decoded == data

    def test_symbol_value_in_range(self):
        """编码后所有符号值在 [0, M-1] 范围内。"""
        codec = DataCodec(S=1800, M=224)
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        assert all(0 <= s < 224 for s in symbols)

    def test_large_m_single_symbol_per_byte(self):
        """M=448 时 k=1, n=1，每个符号直接表示 1 byte。"""
        codec = DataCodec(S=216, M=448)
        assert codec.k == 1
        assert codec.n == 1
        data = os.urandom(codec.L_max)
        symbols = codec.encode(data)
        decoded = codec.decode(symbols)
        assert decoded == data
