# 传输层规范（Transport Layer）

传输层负责将一段已知总长度的字节流（content），用 LT 喷泉码切片、编码为帧、并在接收端重组恢复。传输层不解析 content 内部结构。

**上层**：应用
- 编码方向：接收 content（byte 流）及其长度 total_length
- 解码方向：交付恢复的 content（长度为 total_length）

**下层**：数据层
- 编码方向：交付 byte 块（长度恒为 L_max，不足则传输层 padding）
- 解码方向：接收 byte 块（RS 纠错后，传输层按帧结构解析）；若数据层 RS 解码失败（见 `data_layer.md`"解码失败处理"），本次截图不产生 byte 块，传输层视为未收到该帧，直接跳过

L_max 是数据层的容量属性，传输层查表获取，据此推导 chunk_size（见 2.1 节）。

**约束：一次传输过程中，网格等级与颜色档/图形档必须保持不变（L_max 恒定）。** 原因：chunk_size 由 L_max 推导，后续所有帧的源块切分方案均基于此值；LT 喷泉码的 XOR 运算要求源块长度一致，中途改变 L_max 会导致 chunk_size 变化，等同于重新切分 content，使已发送/接收的帧全部失效。

---

## 1. 帧结构

### Specification

帧格式统一，不区分帧类型。传输层交付/接收的 byte 块（长度恒为 L_max）内部结构：

```
| total_length (4 bytes) | seq (4 bytes) | payload (chunk_size bytes) |
```


- **偏移和长度单位**：byte
- `total_length`：content 总字节数（big-endian），每一帧都携带，接收端收到任何一帧即可获知总长度并引导解码
- `seq`：发送端单调递增分配（big-endian），同一次传输过程中不重复、不归零重置
- 帧长度恒为 L_max（数据层硬性约束），chunk_size 由 L_max 推导（见 2.1 节）
- 不单独传输 seed 字段，degree/indices 推导用的 seed 由 seq 和 total_length 混合得到（见 2.3 节），节省 4 bytes

### Proposal

- 比起单独广播一个元信息帧，把 total_length 每帧都带上能让接收端因为第一帧就开始解码，加入延迟更低

---

## 2. LT 喷泉码

### Specification

- degree 和 indices 不作为字段传输：用 seq 和 total_length（均已在帧内）混合得到 seed，再用确定性算法（PRNG(seed) 播种）依次采样 degree、indices；接收端用相同算法 + 相同 seq/total_length 推导出相同的 degree、indices
- 源块数 K，每块 chunk_size bytes，末块 zero-padded

### 2.1 参数推导

chunk_size、K 不作为字段传输，两端各自推导：

```
chunk_size = L_max − 4(total_length) − 4(seq)   # 帧固定开销之外全部给 payload
K = ceil(total_length / chunk_size)
```

- L_max 由数据层查表得到（给定网格等级 + 颜色档/图形档）
- total_length 从任何一帧的固定字段中获知
- 两端使用相同公式，无需协商

### 2.2 编码过程

```
1. 应用层交付 content，total_length = len(content)
2. chunk_size, K 按 2.1 节公式推导
3. 分块：content 按 chunk_size 切成 K 块，末块 zero-pad
4. 对每个编码块（seq 递增分配）：
   a. degree, indices 按 2.3 节算法从 (seq, total_length) 推导
   b. payload = blocks[indices[0]] XOR ... XOR blocks[indices[degree-1]]
5. 输出 (total_length, seq, payload)
```

### 2.3 确定性算法

编解码两端必须使用逐位一致的算法，否则同一 (seq, total_length) 会推导出不同的 (degree, indices)，导致解码失败。

**seed 派生**：从已传输字段混合得到，避免额外传输，同时防止 seq 偶发重复时与另一个传输会话碰撞：

```
seed = mix32(seq XOR mix32(total_length))
```

**mix32**：采用 splitmix32 finalizer（乘法+移位混合），相比 xorshift 具有更好的雪崩特性，确保相邻输入产生差异足够大的输出：

```
mix32(x):
    x ^= x >> 16
    x *= 0x45d9f3b
    x ^= x >> 16
    x *= 0x45d9f3b
    x ^= x >> 16
    return x & 0xFFFFFFFF
```

**PRNG**：xorshift32，用上述 seed 作为初始状态（seed=0 时状态置为 1，避免退化）：

```
state = (seed == 0) ? 1 : seed
next_u32():
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    return state
```

**degree 采样**（Robust Soliton Distribution，参数 c=0.1，δ=0.5，固定常量）：

```
u = next_u32() / 2^32                     # 映射到 [0, 1)
degree = 满足 CDF_RSD(K, c, δ)(d) >= u 的最小 d   # CDF 由 K 在两端各自构建，公式见标准 RSD 定义
```

**indices 采样**（部分 Fisher-Yates，从 [0, K) 无放回抽 degree 个，继续消耗同一 rng 流）：

```
pool = [0, 1, ..., K-1]
indices = []
for i in 0..degree-1:
    j = next_u32() % (K - i)
    indices.append(pool[j])
    pool[j] = pool[K - i - 1]   # 与末尾元素交换，避免重复
```

### 2.4 解码过程

```
1. 收到一帧：解析 total_length（若首次收到，据此算出 chunk_size, K，初始化 decoder）
2. degree, indices 按 2.3 节算法从 (seq, total_length) 推导（与编码端算法一致，输入相同则结果必然相同）
3. Peeling decoder：
   a. 找 degree=1 的块 → 直接还原对应源块
   b. 用已还原的源块消除其他块中的对应项
   c. 重复直到全部 K 个源块还原
4. 拼接 K 个源块 → 截断至 total_length → 交付应用层 content
```

### Proposal

- 混入 total_length 是为了在 seq 意外重复（如跨会话碰撞）时仍大概率不撞 seed

---

## 3. 应用层接口

具体的 content 内部格式见 [application_layer.md](application_layer.md)，传输层本身不关心。

### Specification

- 编码方向：应用层交付 content（byte 流）及其长度 total_length，传输层原样编码，不解析内部结构
- 解码方向：传输层交付还原后的 content（长度为 total_length），内部结构由应用层解释

---

## 4. 去重

### Specification

- 接收端维护 `seen_seqs` 集合，收到帧后检查 seq 是否已见
- seq 已见 → 丢弃
- seq 未见 → 加入集合，送入 LT decoder

### Proposal

- seq 去重是性能优化：避免重复计算 (degree, indices) 和重复送入 LT decoder
- `seen_seqs` 集合过大时清理旧条目（如保留最近 50000 个）
