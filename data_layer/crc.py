"""
CRC16 校验（CCITT-FALSE 多项式 0x1021，初始值 0xFFFF）

用于 RS 纠错之上的独立完整性校验，防止 RS 误纠正。
"""

from typing import Tuple

_CRC16_POLY = 0x1021
_CRC16_INIT = 0xFFFF


def crc16(data: bytes) -> int:
    """计算 CRC16-CCITT (0x1021, init=0xFFFF, no reflection, no xor-out)。"""
    crc = _CRC16_INIT
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc16_check(data: bytes, expected: int) -> bool:
    """验证 data 的 CRC16 是否等于 expected。"""
    return crc16(data) == expected


def crc16_append(data: bytes) -> bytes:
    """在 data 末尾追加 CRC16（2 bytes, big-endian）。"""
    return data + crc16(data).to_bytes(2, "big")


def crc16_verify(data_with_crc: bytes) -> Tuple[bool, bytes]:
    """验证带 CRC16 的数据。

    返回: (是否校验通过, 去除 CRC 后的数据)
    """
    if len(data_with_crc) < 2:
        return False, b""
    data = data_with_crc[:-2]
    expected = int.from_bytes(data_with_crc[-2:], "big")
    return crc16_check(data, expected), data
