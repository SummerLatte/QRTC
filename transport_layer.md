# 传输层规范（Transport Layer）

传输层负责文件到帧的编码与重组：帧结构定义、LT 喷泉码、payload 字段定义、帧调度、去重、解压。

**上层**：应用
- 编码方向：接收原始文件
- 解码方向：交付恢复的原始文件

**下层**：数据层
- 编码方向：交付 byte 流（长度 L ≤ L_max，不足则 padding）
- 解码方向：接收 byte 流（RS 纠错后，传输层按 payload 语义截断）

L_max 是数据层的容量属性（类比 MTU），传输层查表获取，据此控制 payload 长度。

---

## 1. 帧结构

### Specification

传输层定义帧的封装格式。RS 纠错由数据层统一执行（RS(255, 223)），传输层不介入 RS 参数。

帧分为 Header 帧和 Data 帧，由 1-byte `frame_type` 字段显式区分。封装结构相同：

```
| frame_type (1 byte) | payload | RS 纠错码 |
```

- **偏移和长度单位**：byte
- `frame_type` 作为第一个字段，由 RS 一并保护，避免靠猜内容判断帧类型

### 1.1 frame_type

| 值 | 帧类型 | 说明 |
|----|--------|------|
| 0x00 | Header 帧 | 携带文件元信息，结构固定，周期性广播 |
| 0x01 | Data 帧 | 携带 LT 编码块，结构变长 |

### Proposal

- 显式 `frame_type` 字段由 RS 保护，比靠 magic/seq 猜内容更可靠
- RS 统一参数（RS(255, 223)），数据层自包含执行，传输层不介入
- Header 帧的高可靠靠传输层重复发送（周期性广播），不靠更高 RS 冗余

---

## 2. LT 喷泉码

### Specification

- 数据（可选 zlib 压缩）分块 → RSD 采样 degree → XOR 得到编码块 → 接收端 peeling decoder 恢复
- 源块数 K，每块 chunk_size bytes，末块 zero-padded
- degree 从 Robust Soliton Distribution 采样
- indices 从 [0, K) 中无放回随机抽取 degree 个
- payload = indices 对应源块的 XOR

### 编码过程

```
1. (可选) zlib 压缩
2. 分块：K = ceil(len / chunk_size)，末块 zero-pad
3. 对每个编码块：
   a. 采样 degree ~ RSD(K)
   b. 采样 indices：从 [0, K) 无放回抽取 degree 个
   c. payload = blocks[indices[0]] XOR ... XOR blocks[indices[degree-1]]
4. 输出 (degree, indices, payload)
```

### 解码过程

```
1. 收集编码块 (degree, indices, payload)
2. Peeling decoder：
   a. 找 degree=1 的块 → 直接还原对应源块
   b. 用已还原的源块消除其他块中的对应项
   c. 重复直到全部 K 个源块还原
3. 拼接 K 个源块 → 截断至 filesize → (可选) zlib 解压
```

### Proposal

- LT 喷泉码是无序的：接收端不需要知道"这是第几个包"，只需要 degree + indices + payload
- 喷泉码本身提供冗余：接收端收够略多于 K 个独立编码块即可解码，无需重传

---

## 3. Payload 字段定义

帧结构为 `frame_type + payload + RS`（见第 1 节），传输层定义 payload 内部字段。

偏移和长度单位均为 **byte**。以下偏移相对于 payload 起始位置（不含 frame_type）。

### 3.1 Header payload

| 偏移 | 长度 | 字段 | 说明 |
|------|------|------|------|
| 0 | 1 | content_type | 'F'=文件, 'T'=文本 |
| 1 | 4 | filesize | 原始文件大小 (big-endian) |
| 5 | 4 | total_blocks | LT 码源块数 K |
| 9 | 4 | chunk_size | 每块字节数 |
| 13 | 2 | filename_len | 文件名长度 |
| 15 | 15 | filename | UTF-8 文件名 (截断/填充 0) |
| 30 | ... | padding | 0xFF 填充至 RS 块对齐 |

- Header payload 固定结构，接收端必须先拿到 Header 才能解码任何 Data 帧
- padding 由数据层在 RS 分块时填充

### 3.2 Data payload

| 偏移 | 长度 | 字段 | 说明 |
|------|------|------|------|
| 0 | 4 | seq | 包序号 (big-endian)，用于快速去重 |
| 4 | 2 | degree | XOR 的源块数 |
| 6 | 4×degree | indices | 源块索引 (每个 4 bytes big-endian) |
| 6+4d | chunk_size | payload | XOR 后的编码块数据 |

- seq 由发送端单调递增分配，仅用于接收端快速去重
- LT 解码本身不依赖 seq，只用 degree + indices + payload
- 相同 (degree, indices) 的编码块内容必然相同，seq 去重是性能优化

### Proposal

- seq 保留为快速去重手段：在帧最前面，无需解析完整 payload 即可判断重复
- seq 不参与 LT 解码逻辑，去掉 seq 也能用 (degree, indices) 去重，但解析成本更高

---

## 4. 帧调度

### Specification

广播场景下接收端随时加入，Header 帧周期性重复发送。

- 循环发送：**1 Header + N Data**（N 默认 10）
- Header 帧周期性插入 Data 帧流中
- Data 帧持续生成（LT 喷泉码无限速率），直到接收端解码完成

### Proposal

- Header 周期性广播，类比 DVB-T 的 PSI 表周期性发送
- N 可调：N 太小则有效吞吐低，N 太大则接收端加入延迟高

---

## 5. 去重

### Specification

- 接收端维护 `seen_seqs` 集合，收到 Data 帧后检查 seq 是否已见
- seq 已见 → 丢弃
- seq 未见 → 加入集合，送入 LT decoder

### Proposal

- seq 去重是性能优化：避免重复解析完整 payload 和重复送入 LT decoder
- `seen_seqs` 集合过大时清理旧条目（如保留最近 50000 个）
