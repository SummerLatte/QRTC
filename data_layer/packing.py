"""
多符号联合打包（Multi-Symbol Packing）

将 n bytes 视为一个整数，转换为 k 个 base-M 符号。
约束：M^k >= 256^n（编码无损）。
"""

import math
from typing import List, Tuple

# (k, n) 参数表：给定 M，查表获取打包参数 (k 个符号 ↔ n bytes)
PACKING_TABLE: dict = {
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


def select_packing(M: int) -> Tuple[int, int]:
    """给定 M，选择最优 (k, n) 打包参数。

    优先查表，表外按算法选择效率最高的 (k, n)。
    返回 (k, n)。
    """
    if M in PACKING_TABLE:
        return PACKING_TABLE[M]

    best = None
    for n in range(1, 10):
        k = 1
        while M ** k < 256 ** n:
            k += 1
        eff = (n * 8) / (k * math.log2(M))
        if best is None or eff > best[2]:
            best = (k, n, eff)
    return best[0], best[1]


def pack_bytes_to_symbols(data: bytes, M: int, k: int, n: int) -> List[int]:
    """将 n bytes 转为 k 个 base-M 符号。

    编码方向：bytes → symbols
    """
    assert len(data) == n, f"数据长度 {len(data)} != n={n}"
    value = int.from_bytes(data, "big")
    symbols = [0] * k
    for i in range(k):
        symbols[k - 1 - i] = value % M
        value //= M
    assert value == 0, f"M^k < 256^n, 编码有损! 残余值={value}"
    return symbols


def unpack_symbols_to_bytes(symbols: List[int], M: int, k: int, n: int) -> bytes:
    """将 k 个 base-M 符号转为 n bytes。

    解码方向：symbols → bytes
    """
    value = 0
    for i in range(k):
        value = value * M + symbols[i]
    # 符号值可能因传输错误而越界，取模防止 OverflowError
    max_val = 256 ** n
    value = value % max_val
    return value.to_bytes(n, "big")
