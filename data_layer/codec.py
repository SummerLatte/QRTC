"""
数据层编解码器（DataCodec）

串联多符号打包、RS 纠错、CRC16 校验，实现 byte 块 ↔ 符号块的双向转换。

编码方向（byte → 符号）：
1. 传输层交付 L_max bytes
2. 计算 CRC16(L_max bytes)，拼接得到 L_max+2 bytes
3. RS 编码：(L_max+2) bytes → P × n bytes（R 个块，每块 32 parity，末尾块缩短）
4. 多符号打包：P × n bytes → P × k 个符号
5. 符号填充：若 P × k < S，末尾补 0 符号至 S 个
6. 交付符号层：S 个符号

解码方向（符号 → byte）：
1. 符号层交付 S 个符号 + 置信度
2. 多符号解包：P = floor(S/k) 个打包块 → P × n bytes
   - 含 erasure 的打包块标记对应 n bytes 为 erasure
3. RS 解码：按 R 个块分块，每块纠错输出 data bytes，拼接得 (L_max+2) bytes
4. 验证 CRC16
5. 交付传输层：L_max bytes
"""

import math
from typing import List, Optional, Tuple

from .packing import select_packing, pack_bytes_to_symbols, unpack_symbols_to_bytes
from .rs import rs_encode_block, rs_decode_block
from .crc import crc16, crc16_verify

RS_NSYM = 32          # 每块 32 bytes parity
RS_MAX_DATA = 223     # 每块最大数据 223 bytes (255 - 32)
RS_BLOCK_SIZE = 255   # 满块 255 bytes


class DataCodec:
    """数据层编解码器。

    给定 S（符号容量）和 M（符号总数），自动推导打包参数 (k, n)、
    RS 分块方案和 L_max。
    """

    def __init__(self, S: int, M: int):
        self.S = S
        self.M = M
        self.k, self.n = select_packing(M)

        # 打包块数
        self.P = S // self.k
        # 打包产生的 byte 总数
        self.total_bytes = self.P * self.n
        # RS 块数
        self.R = math.ceil(self.total_bytes / RS_MAX_DATA) if self.total_bytes > 0 else 0
        # parity 总数
        self.parity_total = self.R * RS_NSYM
        # L_max = total_bytes - parity - 2(CRC)
        self.L_max = self.total_bytes - self.parity_total - 2

        if self.L_max < 0:
            raise ValueError(
                f"容量不足: S={S}, M={M} → total_bytes={self.total_bytes}, "
                f"parity={self.parity_total}, L_max={self.L_max}"
            )

    @property
    def packing_params(self) -> Tuple[int, int]:
        """(k, n) 打包参数。"""
        return self.k, self.n

    def encode(self, data: bytes) -> List[int]:
        """编码方向：L_max bytes → S 个符号。

        data 长度必须等于 L_max。
        返回长度为 S 的符号列表（值域 [0, M-1]）。
        """
        if len(data) != self.L_max:
            raise ValueError(f"数据长度 {len(data)} != L_max={self.L_max}")

        # 1. CRC16
        data_with_crc = data + crc16(data).to_bytes(2, "big")
        assert len(data_with_crc) == self.L_max + 2

        # 2. RS 编码：将 total_bytes 分成 R 块，每块最多 255 bytes（含 parity）
        # 每块的数据部分 = block_total_size - 32
        encoded_bytes = bytearray()
        offset = 0
        data_offset = 0
        remaining = self.total_bytes
        for i in range(self.R):
            remaining_blocks = self.R - i
            # 该块总大小（数据+parity），最多 255
            block_total = min(RS_BLOCK_SIZE, remaining - (remaining_blocks - 1) * RS_NSYM)
            block_total = max(RS_NSYM, block_total)  # 至少要有 parity 空间
            block_data_size = block_total - RS_NSYM

            # 从 data_with_crc 中取数据，如果数据不够用 0 填充
            data_end = data_offset + block_data_size
            if data_end <= len(data_with_crc):
                chunk = data_with_crc[data_offset:data_end]
            elif data_offset < len(data_with_crc):
                chunk = data_with_crc[data_offset:] + b"\x00" * (data_end - len(data_with_crc))
            else:
                chunk = b"\x00" * block_data_size

            encoded = rs_encode_block(chunk, RS_NSYM)
            encoded_bytes.extend(encoded)
            data_offset = data_end
            offset += block_total
            remaining -= block_total

        # 确保编码后恰好是 total_bytes
        assert len(encoded_bytes) == self.total_bytes, \
            f"RS 编码后长度 {len(encoded_bytes)} != total_bytes={self.total_bytes}"

        # 3. 多符号打包
        symbols: List[int] = []
        for i in range(self.P):
            chunk = bytes(encoded_bytes[i * self.n:(i + 1) * self.n])
            block_symbols = pack_bytes_to_symbols(chunk, self.M, self.k, self.n)
            symbols.extend(block_symbols)

        # 4. 填充至 S 个符号
        while len(symbols) < self.S:
            symbols.append(0)

        assert len(symbols) == self.S, \
            f"符号数 {len(symbols)} != S={self.S}"
        assert all(0 <= s < self.M for s in symbols), "符号值越界"

        return symbols

    def decode(self, symbols: List[int],
               erasure_flags: Optional[List[bool]] = None) -> Optional[bytes]:
        """解码方向：S 个符号 → L_max bytes。

        symbols: 符号层交付的 S 个符号值。
        erasure_flags: 长度为 S 的布尔列表，True 表示该符号低置信度（erasure）。
            None 表示全部置信。
        返回: L_max bytes（成功时），或 None（RS 或 CRC 校验失败，整帧无效）。
        """
        if len(symbols) != self.S:
            raise ValueError(f"符号数 {len(symbols)} != S={self.S}")

        if erasure_flags is None:
            erasure_flags = [False] * self.S

        # 1. 多符号解包：P 个打包块 → P × n bytes
        # 标记含 erasure 的打包块对应的 n bytes 为 erasure
        raw_bytes = bytearray()
        byte_erasure: List[bool] = []

        for i in range(self.P):
            block_symbols = symbols[i * self.k:(i + 1) * self.k]
            block_erasure = erasure_flags[i * self.k:(i + 1) * self.k]
            # 符号值可能越界（传输错误），取模 M 防止 OverflowError
            safe_symbols = [s % self.M for s in block_symbols]
            chunk = unpack_symbols_to_bytes(safe_symbols, self.M, self.k, self.n)
            raw_bytes.extend(chunk)
            # 整块有 erasure → n bytes 全标记为 erasure
            block_has_erasure = any(block_erasure)
            byte_erasure.extend([block_has_erasure] * self.n)

        assert len(raw_bytes) == self.total_bytes
        assert len(byte_erasure) == self.total_bytes

        # 2. RS 解码：按 R 块分块（与编码端相同的分块方式）
        decoded_bytes = bytearray()
        offset = 0
        remaining = self.total_bytes
        for i in range(self.R):
            remaining_blocks = self.R - i
            block_total = min(RS_BLOCK_SIZE, remaining - (remaining_blocks - 1) * RS_NSYM)
            block_total = max(RS_NSYM, block_total)
            block_data_size = block_total - RS_NSYM

            block_with_parity = bytes(raw_bytes[offset:offset + block_total])

            # 收集该块的 erasure 位置
            block_erasure_pos: List[int] = []
            for j in range(block_total):
                if offset + j < len(byte_erasure) and byte_erasure[offset + j]:
                    block_erasure_pos.append(j)

            corrected, success = rs_decode_block(
                block_with_parity, RS_NSYM,
                erasure_pos=block_erasure_pos if block_erasure_pos else None
            )

            if not success:
                return None

            decoded_bytes.extend(corrected)
            offset += block_total
            remaining -= block_total

        # 3. 验证 CRC16
        data_with_crc = bytes(decoded_bytes)
        if len(data_with_crc) != self.L_max + 2:
            return None

        crc_ok, data = crc16_verify(data_with_crc)
        if not crc_ok:
            return None

        return data

    def __repr__(self) -> str:
        return (
            f"DataCodec(S={self.S}, M={self.M}, k={self.k}, n={self.n}, "
            f"P={self.P}, R={self.R}, L_max={self.L_max})"
        )
