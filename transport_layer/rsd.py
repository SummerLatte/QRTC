"""
Robust Soliton Distribution：CDF 构建 + degree/indices 采样。

参数 c=0.1, δ=0.5（固定常量）。
"""

import math
from typing import Callable, List

from .prng import xorshift32


def robust_soliton_cdf(K: int, c: float = 0.1, delta: float = 0.5) -> List[float]:
    """构建 RSD 的 CDF。

    返回长度 K+1 的列表，cdf[d] = P(degree <= d)，d 从 0 到 K。
    cdf[0] = 0.0。
    """
    if K <= 0:
        return [0.0]

    S = c * math.log(K / delta) * math.sqrt(K)
    S = math.ceil(S)
    if S <= 0:
        S = 1

    # Ideal Soliton distribution
    tau = [0.0] * (K + 1)
    tau[1] = 1.0 / K
    for d in range(2, K + 1):
        tau[d] = 1.0 / (d * (d - 1))

    # Robust component
    rho = [0.0] * (K + 1)
    K_S = K // S if S > 0 else K
    if K_S < 1:
        K_S = 1
    for d in range(1, K_S):
        rho[d] = S / (K * d)
    if K_S <= K:
        rho[K_S] = S * math.log(S / delta) / K if S > 0 else 0.0

    # Combine and normalize
    Z = sum(tau[d] + rho[d] for d in range(1, K + 1))
    if Z <= 0:
        Z = 1.0

    cdf = [0.0] * (K + 1)
    cum = 0.0
    for d in range(1, K + 1):
        cum += (tau[d] + rho[d]) / Z
        cdf[d] = cum
    return cdf


def sample_degree(rng: Callable[[], int], K: int, cdf: List[float]) -> int:
    """从 RSD 采样 degree。"""
    u = rng() / (2 ** 32)
    for d in range(1, K + 1):
        if cdf[d] >= u:
            return d
    return K


def sample_indices(rng: Callable[[], int], K: int, degree: int) -> List[int]:
    """部分 Fisher-Yates 采样：从 [0, K) 无放回抽 degree 个。"""
    pool = list(range(K))
    indices = []
    for i in range(degree):
        j = rng() % (K - i)
        indices.append(pool[j])
        pool[j] = pool[K - i - 1]
    return indices


def derive_degree_indices(seq: int, total_length: int, K: int,
                          cdf: List[float]) -> tuple:
    """从 (seq, total_length) 推导 (degree, indices)。"""
    from .prng import derive_seed
    seed = derive_seed(seq, total_length)
    rng = xorshift32(seed)
    degree = sample_degree(rng, K, cdf)
    indices = sample_indices(rng, K, degree)
    return degree, indices
