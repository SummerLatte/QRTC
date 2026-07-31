"""
Format Info 子系统：BCH(14,4) 编解码。

编码颜色档位和图形档位，供解码端确定符号空间。
4 data bits + 10 BCH 纠错 bits = 14 bits
冗余存放两份，共 28 个黑白模块。

| Bit | 含义 |
|-----|------|
| 0-1 | 颜色档位：00=2色, 01=4色, 10=8色 |
| 2-3 | 图形档位：00=2, 01=4, 10=8, 11=16 |
| 4-13 | BCH(14,4) 纠错位 |
"""

from __future__ import annotations

from dataclasses import dataclass


# BCH(14,4) 参数
# 生成多项式 g(x) = (x+1)(x^3+x+1)^2(x^3+x^2+1) = x^10 + x^8 + x^7 + x^3 + x + 1
# 对应二进制: 0b10110001011 = 0x58B
_BCH_GEN = 0x58B       # 生成多项式（含 x^10 最高位）
_BCH_GEN_DEGREE = 10
_BCH_N = 14
_BCH_K = 4
_BCH_T = 10  # 纠错位数


def _gf2_mod(dividend: int) -> int:
    """GF(2) 多项式取模：dividend mod g(x)。"""
    result = dividend
    while result.bit_length() > _BCH_GEN_DEGREE:
        shift = result.bit_length() - 1 - _BCH_GEN_DEGREE
        result ^= _BCH_GEN << shift
    return result


def _bch_encode(data_bits: int) -> int:
    """
    BCH(14,4) 编码：4 data bits → 14 bits。

    data_bits: 低 4 位为数据
    返回: 14 bits，高 4 位为数据，低 10 位为纠错位
    """
    data = data_bits & 0xF
    msg = data << _BCH_T           # 数据左移 10 位
    remainder = _gf2_mod(msg)      # 求余数
    return msg | remainder


def _bch_decode(codeword: int) -> tuple[int, int]:
    """
    BCH(14,4) 解码：14 bits → (data_bits, error_count)。

    返回 (data, error_count)，error_count=0 表示无错。
    能纠正 1 位错误。
    """
    codeword &= 0x3FFF  # 14 bits
    syndrome = _gf2_mod(codeword)

    if syndrome == 0:
        return (codeword >> _BCH_T) & 0xF, 0

    # 查找单错误位置
    for pos in range(_BCH_N):
        if _gf2_mod(1 << pos) == syndrome:
            corrected = codeword ^ (1 << pos)
            return (corrected >> _BCH_T) & 0xF, 1

    # 多位错误，无法纠正
    return (codeword >> _BCH_T) & 0xF, -1


@dataclass(frozen=True)
class FormatInfo:
    """Format Info 数据。"""
    color_level_code: int  # 2 bits: 颜色档位编码
    shape_level_code: int  # 2 bits: 图形档位编码

    @property
    def data_bits(self) -> int:
        """4 data bits: [color(2)] [shape(2)]。"""
        return ((self.color_level_code & 0x3) << 2) | (self.shape_level_code & 0x3)

    def encode(self) -> int:
        """编码为 14-bit codeword。"""
        return _bch_encode(self.data_bits)

    def to_bit_list(self) -> list[int]:
        """转为 14 个 bit 的列表（bit 0 在前）。"""
        cw = self.encode()
        return [(cw >> i) & 1 for i in range(14)]

    @staticmethod
    def from_codes(color_level_code: int, shape_level_code: int) -> "FormatInfo":
        return FormatInfo(color_level_code, shape_level_code)

    @staticmethod
    def decode(codeword: int) -> tuple["FormatInfo", int]:
        """
        从 14-bit codeword 解码。

        返回 (FormatInfo, error_count)。
        error_count=0 无错, 1 已纠正, -1 多位错误不可纠正。
        """
        data, err = _bch_decode(codeword)
        color_code = (data >> 2) & 0x3
        shape_code = data & 0x3
        return FormatInfo(color_code, shape_code), err

    @staticmethod
    def from_bit_list(bits: list[int]) -> tuple["FormatInfo", int]:
        """从 14 个 bit 的列表解码（bit 0 在前）。"""
        if len(bits) != 14:
            raise ValueError(f"Format Info 需要 14 bits, 得到 {len(bits)}")
        cw = sum(b << i for i, b in enumerate(bits))
        return FormatInfo.decode(cw)
