# 数据层规范（Data Layer）

数据层负责符号块与字节块之间的双向转换：多符号联合打包、RS 纠错、CRC16 完整性校验。

数据层只接收 byte 块，输出符号块（编码方向），或接收符号块输出 byte 块（解码方向）。

**上层**：传输层
- 编码方向：接收 byte 块（长度恒为 L_max），打包为符号块
- 解码方向：交付 byte 块（RS 纠错后）

**下层**：符号层
- 编码方向：交付符号块（恰好 S 个符号，每个值 0 到 M-1）
- 解码方向：接收符号块（S 个）+ 置信度（erasure 标记）

**容量参数**：
- 符号层公布 S（单帧符号容量，由网格等级决定）和 M（符号总数，由颜色档/图形档决定）
- 数据层公布 L_max（单帧可承载的 data byte 容量，由 S、M、打包参数决定，RS 参数固定）
- 传输层查 L_max，据此控制 payload 长度

---

## 0. 符号

### Specification

- **符号（Symbol）** 是一个非负整数，值域为 [0, M-1]
- M 为符号总数（符号空间的基数），由符号层决定
- 数据层只接收 M 值，不感知 M 的来源
- 符号块（symbol block）是 S 个符号的有序序列

---

## 0.1 容量

### Specification

数据层的容量 L_max（单帧可承载的 data byte 数）由 S、M、打包参数 (k, n) 决定，RS 参数固定为 (255, 223, 32 parity)。给定 S 和 M 后 L_max 为固定常量。

```
打包块数 P = floor(S / k)
打包产生 byte 数 = P × n
RS 块数 R = ceil(P × n / 255)
L_max = P × n − R × 32 − 2   # 末尾 2 bytes 为 CRC16，RS 保护范围内，不计入可用容量
```

- 每个 RS 块固定 32 bytes parity，数据部分最多 223 bytes
- 末尾块使用缩短 RS（shortened RS）：数据部分可少于 223 bytes，parity 不变
- R = ceil 而非 floor：充分利用末尾空间，不浪费整块
- CRC16（2 bytes）覆盖 RS 纠正后的 L_max bytes，详见"CRC 校验"一节

约束：
- 传输层交付的 byte 块长度恒为 L_max（原始 payload 不足时由传输层 padding 补齐）
- S 不一定能被 k 整除：末尾不足 k 个的符号填充固定值（如 0）

---

## 1. 多符号联合打包

### Specification

- 将 n bytes 视为一个整数，转换为 k 个 base-M 符号（RS 输出的 bytes 需转成符号才能显示）
- M 通常不是 2 的幂，无法直接按 bit 对齐，联合打包提升 byte 利用率
- 约束：M^k ≥ 256^n（k 个符号必须能表示 n bytes 的全部取值，否则编码有损）
- 给定 M，(k, n) 由下表确定

### 打包规则（byte → 符号，编码方向）

```
value = bytes_to_int(n bytes)
for i in 0..k-1:
    s[k-1-i] = value % M
    value = value // M
```

### 解包规则（符号 → byte，解码方向）

```
value = s[0] × M^(k-1) + s[1] × M^(k-2) + ... + s[k-1] × M^0
value 转为 n bytes（big-endian）
```

### (k, n) 参数表

| M | k（符号数） | n（bytes） |
|---|------------|-----------|
| 2 | 8 | 1 |
| 4 | 4 | 1 |
| 8 | 8 | 3 |
| 12 | 7 | 3 |
| 16 | 2 | 1 |
| 24 | 2 | 1 |
| 48 | 3 | 2 |
| 56 | 3 | 2 |
| 96 | 4 | 3 |
| 112 | 4 | 3 |
| 224 | 4 | 3 |
| 448 | 1 | 1 |

- 效率 = (n × 8) / (k × log2(M))，恒 ≤ 1，M = 2^m 时可取到 100%

### Proposal

- 多符号联合打包而非逐符号转 bits，是因为 M 非 2 的幂时逐符号转换浪费大
- 对每个候选 n，取满足 M^k ≥ 256^n 的最小 k，再从中选效率最高的 (k, n)
- 单个符号错误会扩散到所在打包块的 n bytes，但 RS 按 byte 纠错，影响可控

---

## 2. Reed-Solomon 纠错

### Specification

- RS 码工作在 GF(256) 上，每个 RS 符号 = 1 byte
- 打包后的 byte 块按 RS(255, 223) 分块纠错，统一 parity 长度（32 bytes/块）
- 支持缩短 RS（shortened RS）：末尾块数据部分可少于 223 bytes，parity 不变
- 数据层根据已知 byte 长度自行分块，无需外部参数

| 参数 | 值 |
|------|-----|
| RS 码 | RS(255, 223) on GF(256)，支持缩短 |
| 纠错能力 | 每块纠正 16 bytes error 或 32 bytes erasure |
| 纠错码占比 | 满块 ~14%，缩短块略高 |

### erasure 机制

- 符号层可输出符号置信度，低置信度符号标记为 erasure（不依赖值范围判断）
- 打包块内含 erasure 符号 → 整块标记为 erasure（n bytes 位置已知）
- RS 纠错：erasure 1:1（1 个纠错 byte 纠 1 个 erasure byte），error 1:2
- erasure 使 RS 纠错能力翻倍

### CRC 校验（RS 之上的独立完整性把关）

- RS 解码成功后，对纠正后的 L_max bytes 再做一次 CRC16 校验（2 bytes，编码时附加在 RS 保护范围内，解码时校验后丢弃，不计入 L_max 的可用容量）
- 这是网络协议的通用模式（WiFi/LTE/DVB 等均在 FEC 之上叠加独立 CRC）：不单纯依赖 RS 解码器自身的成功/失败信号，因为纠错码存在极小概率的"误纠正"（miscorrection，错误数超限但恰好收敛到另一合法码字，RS 报告成功但内容实际错误）
- CRC 成本低（2 bytes/帧）但能同时防住 RS 误纠正和潜在实现 bug，性价比高，故直接采纳

### 解码失败处理

- 任一 RS 块解码失败，或 RS 成功但 CRC16 校验不过 → 整个符号块判定为**无效帧**，不生成 byte 块，不交付传输层
- 无效帧对传输层等同于"这一帧没收到"：不参与 LT decoder，不消耗 seq 去重资源

---

## 3. 处理顺序

### 编码方向（byte → 符号）

```
1. 传输层交付 L_max bytes
2. 计算 CRC16(L_max bytes)，拼接得到 L_max+2 bytes
3. RS 编码：(L_max+2) bytes → P × n bytes（R 个块，每块 32 parity，末尾块缩短）
4. 多符号打包：P × n bytes → P × k 个符号
5. 符号填充：若 P × k < S，末尾补 0 符号至 S 个
6. 交付符号层：S 个符号
```

### 解码方向（符号 → byte）

```
1. 符号层交付 S 个符号 + 置信度
2. 多符号解包：P = floor(S/k) 个打包块 → P × n bytes
   - 含 erasure 的打包块标记对应 n bytes 为 erasure
3. RS 解码：按 R 个块分块（末尾块缩短），每块纠错输出 data bytes，拼接得 (L_max+2) bytes
   - 任一块解码失败 → 整帧判定无效，终止，不产生输出（见"解码失败处理"）
4. 验证 CRC16：前 L_max bytes 的 CRC16 与末尾 2 bytes 比对，不一致 → 整帧判定无效，终止
5. 交付传输层：L_max bytes（传输层根据 payload 语义截断至原始 L）
```

---

## 附录

### 容量示例（M=24, k=2, n=1）

| S | P=floor(S/2) | P×n (bytes) | R=ceil(P×n/255) | parity=R×32 | L_max=P×n−R×32−2 |
|-------------|-------------|-------------|-----------------|-------------|------------------|
| 211 | 105 | 105 | 1 | 32 | 71 |
| 595 | 297 | 297 | 2 | 64 | 231 |
| 1107 | 553 | 553 | 3 | 96 | 455 |
| 1747 | 873 | 873 | 4 | 128 | 743 |

> 使用缩短 RS（shortened RS）：末尾块数据部分可少于 223 bytes，parity 仍为 32 bytes。S=211 时使用 RS(105, 73)，所有 S 值均可正常工作。
> 纠错能力不变：每块仍可纠正 16 bytes error 或 32 bytes erasure，缩短块的纠错比例反而更高。

### (k, n) 选择算法

给定 M，遍历小的 n，每个 n 下取满足 M^k ≥ 256^n（编码无损）的最小 k，再从中选效率最高的 (k, n)：

```
candidates = []
for n in 1, 2, 3, ...:
    k = 1
    while M^k < 256^n:
        k += 1
    efficiency = (n * 8) / (k * log2(M))
    candidates.append((k, n, efficiency))
return max(candidates, key=efficiency)
```

- 效率 = (n × 8) / (k × log2(M))，恒 ≤ 1，表示 k 个符号的表示能力被 n bytes 实际利用的比例
- M = 2^m 时存在 (k, n) 使 M^k = 256^n，效率恰好 100%
- M 非 2 的幂时效率 < 100%，选择最接近 100% 的 (k, n)
