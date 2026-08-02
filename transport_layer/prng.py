"""
确定性 PRNG：mix32、xorshift32、seed 派生。

编解码两端必须使用逐位一致的算法。
"""

from typing import Callable


def mix32(x: int) -> int:
    """splitmix32 finalizer：良好雪崩特性，相邻输入输出差异大。"""
    x &= 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x45d9f3b) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x45d9f3b) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x


def derive_seed(seq: int, total_length: int) -> int:
    """seed = mix32(seq XOR mix32(total_length))。"""
    return mix32((seq ^ mix32(total_length)) & 0xFFFFFFFF)


def xorshift32(seed: int) -> Callable[[], int]:
    """xorshift32 PRNG，返回 next_u32 闭包。

    seed=0 时状态置为 1，避免退化。
    """
    state = 1 if seed == 0 else seed & 0xFFFFFFFF

    def next_u32() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17)
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    return next_u32
