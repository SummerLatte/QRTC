"""
Reed-Solomon 纠错：RS(255, 223) on GF(256)，支持缩短 RS 和 erasure。

- 每块 32 bytes parity，最多纠正 16 bytes error 或 32 bytes erasure
- 支持缩短 RS：数据部分可少于 223 bytes，parity 不变
- erasure 位置已知时纠错能力翻倍（1:1 vs error 的 1:2）

底层使用 reedsolo 库实现 BM/Chien/Forney 算法。
"""

from typing import List, Optional, Tuple

import reedsolo


def rs_encode_block(data: bytes, nsym: int = 32) -> bytes:
    """RS 编码一个块。

    data: 数据 bytes（长度 <= 223）
    nsym: parity 字节数（默认 32）
    返回: data + parity（总长度 = len(data) + nsym）
    """
    if len(data) == 0:
        # reedsolo 对空数据不生成 parity，直接返回全零 parity
        return b"\x00" * nsym
    rs = reedsolo.RSCodec(nsym=nsym, fcr=0, prim=0x11d, c_exp=8)
    encoded = rs.encode(bytearray(data))
    # reedsolo 返回 (data, parity) 拼接的 bytearray
    return bytes(encoded)


def rs_decode_block(data_with_parity: bytes, nsym: int = 32,
                    erasure_pos: Optional[List[int]] = None) -> Tuple[bytes, bool]:
    """RS 解码一个块。

    data_with_parity: 数据 + parity bytes
    nsym: parity 字节数（默认 32）
    erasure_pos: 已知错误位置列表（从 0 开始的 byte 索引，0 = 第一个 byte）
    返回: (纠正后的数据 bytes, 是否成功)
    """
    if erasure_pos is None:
        erasure_pos = []

    # 过滤越界位置
    nmess = len(data_with_parity)
    erasure_pos = sorted([p for p in erasure_pos if 0 <= p < nmess])

    rs = reedsolo.RSCodec(nsym=nsym, fcr=0, prim=0x11d, c_exp=8)

    try:
        result = rs.decode(bytearray(data_with_parity), erase_pos=erasure_pos)
        # reedsolo 返回 (decoded_data, decoded_msgecc, errata_pos)
        if isinstance(result, tuple):
            decoded = bytes(result[0])
        else:
            decoded = bytes(result)

        # 验证长度
        expected_data_len = nmess - nsym
        if len(decoded) != expected_data_len:
            # reedsolo 可能返回完整消息，截取数据部分
            decoded = decoded[:expected_data_len]

        return decoded, True
    except (reedsolo.ReedSolomonError, Exception):
        return data_with_parity[:nmess - nsym], False

