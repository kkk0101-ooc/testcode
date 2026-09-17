# 八、KV Cache 与长上下文基础原理

[返回学习列表](学习列表.md#learning-list-kv-cache-long-context)

目标：理解 Decoder-only 模型推理中的 Prefill 与 Decode，能推导每层 KV Cache 的张量形状、容量和计算收益，并能从 Query Head 与 KV Head 的关系解释 MHA、GQA、MQA。

本章边界：只讨论模型层面的基本原理。PagedAttention、Continuous Batching、缓存量化、并行切分、显存碎片与服务吞吐调优放在“大模型推理与分布式系统”模块。

统一记号：

~~~text
B:      batch size
L:      Transformer 层数
D:      hidden size
Tp:     Prompt 的 Token 数
T:      当前已缓存的上下文长度
Hq:     Query Head 数
Hkv:    Key/Value Head 数
Dh:     每个 Attention Head 的维度
g:      每组 Query Head 数，g = Hq / Hkv
s:      每个缓存元素的字节数，例如 FP16/BF16 为 2
~~~

常见结构满足：

$$
D=H_qd_h,
\qquad
H_q\bmod H_{kv}=0
$$

单层缓存的语义形状为：

$$
K_{cache}^{(l)},V_{cache}^{(l)}
\in
\mathbb{R}^{B\times H_{kv}\times T\times d_h}
$$

只计算 K、V 本身时，整个模型的理论 KV Cache 大小为：

$$
M_{KV}
=
2LBTH_{kv}d_hs
\quad\text{bytes}
$$

其中最前面的 $2$ 分别对应 Key 和 Value。

## 知识点导航

- [1. Prefill 和 Decode](#kv-1-prefill-decode)
- [2. KV Cache 保存了哪些张量](#kv-2-cache-tensors)
- [3. KV Cache 为什么能减少重复计算](#kv-3-reuse)
- [4. 单层 K、V 的张量形状](#kv-4-shapes)
- [5. 序列长度与 Attention 计算、显存](#kv-5-long-sequence-cost)
- [6. MHA、GQA、MQA 的结构差异](#kv-6-mha-mqa-gqa)
- [7. MQA/GQA 为什么减少缓存](#kv-7-cache-reduction)
- [8. 长上下文不等于有效利用全部信息](#kv-8-effective-context)
- [9. 完整实践任务](#kv-9-practice)
- [10. 常见错误](#kv-10-common-errors)
- [11. 本章验收](#kv-11-checklist)

---

<a id="kv-1-prefill-decode"></a>

## 1. Prefill 和 Decode

Decoder-only 模型生成可以分成两个阶段：

~~~text
Prompt Token x1:T_p
        |
        v
Prefill：一次处理整个 Prompt，建立各层 KV Cache
        |
        v
用最后位置 Logits 选出 y1
        |
        v
Decode：输入 y1，读取历史 Cache，追加 y1 的 K/V，选出 y2
        |
        v
Decode：输入 y2，读取历史 Cache，追加 y2 的 K/V，选出 y3
        |
       ...
~~~

### 1.1 Prefill

给定提示 Token：

$$
x_{1:T_p}=(x_1,x_2,\ldots,x_{T_p})
$$

Prefill 将整个 Prompt 送入模型：

~~~text
input_ids: [B,Tp]
hidden:    [B,Tp,D]
logits:    [B,Tp,Vocab]
~~~

在每一层，模型为 Prompt 的全部位置计算 Q、K、V。生成时需要保存该层所有 Prompt 位置的 K、V：

~~~text
K_cache[l]: [B,Hkv,Tp,Dh]
V_cache[l]: [B,Hkv,Tp,Dh]
~~~

Prefill 输出最后一个 Prompt 位置的 Logits：

$$
z_{T_p}=f_\theta(x_{1:T_p})[-1]
$$

再由 Greedy、Top-k、Top-p 等规则选出第一个生成 Token $y_1$。

Prefill 内部的 Prompt 位置可以在 Causal Mask 下并行计算，因为真实 Prompt 已经全部给定。并行计算不表示前面的位置可以读取未来位置；Causal Mask 仍限制第 $i$ 个位置只能读取 $1,\ldots,i$。

### 1.2 Decode

选出 $y_1$ 后，下一次模型调用只需要输入这个新 Token，并传入 Prefill 得到的历史缓存：

~~~text
new input_ids: [B,1]
past K/V:      [B,Hkv,Tp,Dh]
new Q/K/V:     Query 为 [B,Hq,1,Dh]，Key/Value 为 [B,Hkv,1,Dh]
updated K/V:   [B,Hkv,Tp+1,Dh]
new logits:    [B,1,Vocab]
~~~

此时新 Query 读取 Prompt 的 K/V 和 $y_1$ 的新 K/V，得到用于选择 $y_2$ 的 Logits。之后每轮重复相同过程。

第 $t$ 个生成 Token 仍满足：

$$
y_t\sim p_\theta(\cdot\mid x_{1:T_p},y_{<t})
$$

KV Cache 只优化每一步如何计算这个条件分布，不改变自回归分解，也不能让同一条序列的所有未来 Token 同时生成。

### 1.3 一个容易混淆的长度关系

Prefill 后：

~~~text
Cache 已包含：Prompt x1:T_p
最后位置 Logits 用于选择：y1
Cache 尚未包含：y1
~~~

把 $y_1$ 作为下一次 Decode 的输入后：

~~~text
Cache 已包含：Prompt x1:T_p + y1
本次 Logits 用于选择：y2
~~~

所以“已选出的 Token 数”和“已经写入 Cache 的生成 Token 数”在模型调用边界上可能相差 1。检查缓存长度时必须说明是在调用前还是调用后观察。

### 1.4 两阶段对比

| 维度 | Prefill | Decode |
|---|---|---|
| 一次调用的新输入长度 | 通常为整个 Prompt 长度 $T_p$ | 通常为 1 |
| Query 长度 | $T_p$ | 1 |
| Key/Value 可见长度 | $T_p$ | 历史长度加当前 Token |
| 位置计算 | Prompt 内可并行 | 单条序列跨生成步串行 |
| 主要输出 | 首个生成 Token 的 Logits、初始 Cache | 下一个 Token 的 Logits、增长后的 Cache |
| Attention 主要计算 | 全 Prompt 的 Causal Self-Attention | 一个新 Query 读取全部历史 K/V |

---

<a id="kv-2-cache-tensors"></a>

## 2. KV Cache 保存了哪些张量

### 2.1 每层 Self-Attention 的 Key 和 Value

第 $l$ 层输入为 $X^{(l)}$。忽略 Bias 时：

$$
Q^{(l)}=X^{(l)}W_Q^{(l)}
$$

$$
K^{(l)}=X^{(l)}W_K^{(l)}
$$

$$
V^{(l)}=X^{(l)}W_V^{(l)}
$$

模型要为每个 Transformer 层分别保存该层历史位置的：

$$
\operatorname{Cache}^{(l)}
=
\left(K_{1:T}^{(l)},V_{1:T}^{(l)}\right)
$$

不能只保存第一层或最后一层的 K、V，因为第 $l$ 层的新 Query 必须读取同一层历史 Token 的 K、V。

### 2.2 为什么不缓存历史 Query

位置 $i$ 的 Query 只在计算位置 $i$ 的输出时使用：

$$
O_i
=
\operatorname{softmax}
\left(
\frac{Q_iK_{\le i}^{T}}{\sqrt{d_h}}
\right)
V_{\le i}
$$

未来位置 $j>i$ 到来时，需要的是新的 $Q_j$ 与历史的 $K_{\le j},V_{\le j}$：

$$
O_j
=
\operatorname{softmax}
\left(
\frac{Q_jK_{\le j}^{T}}{\sqrt{d_h}}
\right)
V_{\le j}
$$

历史 $Q_i$ 不再参与未来位置 $j$ 的 Attention，因此标准 KV Cache 不保存历史 Query。

### 2.3 通常不属于 KV Cache 的内容

标准 Decoder-only Self-Attention 的 KV Cache 不等于“把所有中间结果都保存下来”。通常不包括：

- 历史 Query；
- 历史 Attention Score 或 Softmax 权重；
- 每层所有历史 Hidden State；
- 历史 Logits 或概率分布；
- Token 文本本身。

推理框架还可能维护位置索引、有效长度、页表或其他元数据，但这些不是 K、V 两个核心张量。不同框架的缓存对象结构不同，不能把某个 Python 容器的字段布局当成模型数学定义。

### 2.4 RoPE 与缓存内容

若模型对 Q、K 使用 RoPE，则位置旋转发生在 Attention 点积之前：

$$
\widetilde Q_i=R_iQ_i,
\qquad
\widetilde K_i=R_iK_i
$$

后续位置需要直接用带有正确位置信息的历史 Key 做点积，因此常见实现缓存已经应用对应位置旋转后的 $\widetilde K_i$；Value 不做 RoPE。具体代码可能因实现而异，但必须保证每个缓存 Key 与原 Token 位置一致。

这也是使用 Cache 时必须正确传递 Position ID 或 Cache Position 的原因。若把新 Token 错当成位置 0，形状可能仍合法，语义却已经错误。

### 2.5 为什么通常不在训练中依赖 KV Cache

训练时整段真实 Token 已知，可在 Causal Mask 下并行得到所有位置输出，并且反向传播需要完整计算图。自回归推理没有未来 Token，才需要一轮一轮地调用模型并复用历史 K、V。

因此：

~~~text
训练位置并行 + Causal Mask：解决训练计算方式。
推理 KV Cache：解决跨生成步的历史重复计算。
~~~

二者不是互相替代的开关。

---

<a id="kv-3-reuse"></a>

## 3. KV Cache 为什么能减少重复计算

### 3.1 不使用 Cache

假设 Prompt 为 $[x_1,x_2,x_3]$，已经生成 $y_1,y_2$。不使用 Cache 时，每一步都重新输入完整上下文：

~~~text
第 1 步：[x1,x2,x3]       -> 选 y1
第 2 步：[x1,x2,x3,y1]    -> 选 y2
第 3 步：[x1,x2,x3,y1,y2] -> 选 y3
~~~

虽然每一步只消费最后位置的 Logits，普通完整前向仍会重新计算所有可见位置在每一层的 Hidden State、Q、K、V、Attention 和 FFN。历史 Token 的 K、V 被一遍遍算出。

当前上下文长度为 $T$ 时，完整 Self-Attention 的主要计算为：

$$
O(BT^2D)
$$

投影和 FFN 等稠密层计算约为：

$$
O(BTD^2)
$$

### 3.2 使用 Cache

固定模型参数并处于推理模式时，历史 Token 在同一层已经得到的 K、V 不会因为右侧追加新 Token 而改变。Causal Mask 保证历史位置原本也看不到未来 Token。

因此 Decode 只计算当前新 Token 在每层的：

~~~text
当前 Hidden State
-> 当前 Q、K、V
-> 当前 Q 读取历史 K/V 与当前 K/V
-> 当前 Attention 输出
-> 当前 FFN 输出
-> 把当前 K/V 追加到本层 Cache
~~~

在历史长度约为 $T$ 时，单个新 Token 的 Attention 主要计算降为：

$$
O(BTD)
$$

当前 Token 的稠密层计算约为：

$$
O(BD^2)
$$

所有层都要执行上述过程，因此完整模型还要乘以层数 $L$。

### 3.3 同一步计算对比

| 当前长度约为 $T$ | 不使用 KV Cache | 使用 KV Cache Decode |
|---|---:|---:|
| 本次输入位置数 | $T$ | 1 |
| Query 长度 | $T$ | 1 |
| Key/Value 长度 | $T$ | $T$ |
| Attention 主要计算 | $O(BT^2D)$ | $O(BTD)$ |
| 稠密层主要计算 | $O(BTD^2)$ | $O(BD^2)$ |
| 是否重算历史 K/V | 是 | 否 |
| 持久缓存 | 无 | 随 $T$ 线性增长 |

KV Cache 是典型的“以空间换时间”：增加持久显存占用，避免重复计算历史位置。

### 3.4 KV Cache 没有消除什么

即使使用 Cache：

- 当前 Token 的 Query 仍要读取全部可见历史 Key；
- Attention Decode 计算和内存读取仍随上下文长度 $T$ 线性增加；
- 当前 Token 仍要通过所有 Transformer 层；
- 生成步之间仍有严格的自回归依赖；
- Prompt 的 Prefill 仍要处理完整的 Prompt Attention。

所以 KV Cache 不会把长上下文解码变成与序列长度完全无关的常数开销。

### 3.5 开启和关闭 Cache 是否应改变结果

在以下条件相同时，两种计算路径在数学上应等价：

~~~text
相同模型参数
model.eval()
无随机 Dropout
相同 Token 与 Attention Mask
相同 Position ID / Cache Position
相同解码规则
~~~

浮点运算顺序或不同 Attention Kernel 可能产生很小的数值差异；极少数接近并列的 Logits 可能因此改变最终 Token。若差异明显，应优先检查位置、Mask、缓存长度和输入 Token 是否对齐，而不是认为 KV Cache 改变了模型定义。

---

<a id="kv-4-shapes"></a>

## 4. 单层 K、V 的张量形状

### 4.1 从投影矩阵推导

单层输入：

$$
X\in\mathbb{R}^{B\times T\times D}
$$

允许 Query Head 数和 KV Head 数不同时，投影矩阵为：

$$
W_Q\in\mathbb{R}^{D\times(H_qd_h)}
$$

$$
W_K,W_V\in\mathbb{R}^{D\times(H_{kv}d_h)}
$$

线性投影并拆头之前：

~~~text
Q_linear: [B,T,Hq*Dh]
K_linear: [B,T,Hkv*Dh]
V_linear: [B,T,Hkv*Dh]
~~~

拆头并换轴后，语义形状为：

~~~text
Q: [B,Hq,T,Dh]
K: [B,Hkv,T,Dh]
V: [B,Hkv,T,Dh]
~~~

因此单层、单个张量的元素数为：

$$
N_K=N_V=BTH_{kv}d_h
$$

K 与 V 合计：

$$
N_{K+V}=2BTH_{kv}d_h
$$

### 4.2 Decode 时的形状

已有 $T_{past}$ 个缓存 Token，当前输入 1 个 Token：

~~~text
Q_new:        [B,Hq,1,Dh]
K_new/V_new:  [B,Hkv,1,Dh]
K_cache_old:  [B,Hkv,Tpast,Dh]
K_cache_new:  [B,Hkv,Tpast+1,Dh]
V_cache_new:  [B,Hkv,Tpast+1,Dh]
~~~

当 $H_{kv}<H_q$ 时，每组 Query Head 共享同一个 KV Head。令：

$$
g=\frac{H_q}{H_{kv}}
$$

第 $i$ 个 Query Head 使用的 KV Head 可写为：

$$
j=\left\lfloor\frac{i}{g}\right\rfloor
$$

逻辑上的 Attention Score 仍为：

~~~text
scores: [B,Hq,1,Tpast+1]
output: [B,Hq,1,Dh]
~~~

实现可以通过广播、索引或专用 Kernel 让多个 Query Head 读取共享 K/V，不要求真的把缓存复制成 $H_q$ 份。若先物理执行 Repeat 再保存，便失去了 GQA/MQA 的缓存收益。

### 4.3 物理布局与语义形状

本章统一写成：

~~~text
[B,Hkv,T,Dh]
~~~

但实际库也可能使用：

~~~text
[B,T,Hkv,Dh]
Cache 类对象
每层一个 (K,V) 元组
预分配的更长缓存张量加有效长度
~~~

判断形状时应先确认每个轴的语义，不能只凭下标位置猜测。对于 Hugging Face 常见 Legacy Cache，单层 K、V 通常是 <code>[B,Hkv,T,Dh]</code>。

### 4.4 从模型配置读取形状参数

常见配置关系：

~~~python
Hq = config.num_attention_heads
Hkv = getattr(config, "num_key_value_heads", Hq)
Dh = getattr(config, "head_dim", config.hidden_size // Hq)
L = config.num_hidden_layers
~~~

不能总是假定 $H_{kv}=H_q$。对 GQA/MQA 模型若错误地使用 Query Head 数估算缓存，会把结果高估为：

$$
\frac{H_q}{H_{kv}}
$$

倍。

---

<a id="kv-5-long-sequence-cost"></a>

## 5. 序列长度与 Attention 计算、显存

“序列变长会增加开销”必须区分 Prefill、Decode、临时激活和持久 KV Cache。

### 5.1 Prefill 的 Attention 计算

Prompt 长度为 $T$ 时，每个 Query 位置最多读取 $T$ 个 Key 位置。所有头的 Attention 计算主要为：

$$
QK^T:
O(BH_qT^2d_h)
=
O(BT^2D)
$$

$$
AV:
O(BH_qT^2d_h)
=
O(BT^2D)
$$

Causal Mask 使约一半位置不可见，但稠密实现的渐近复杂度仍是 $O(T^2)$。

概念上的 Attention Score 形状为：

$$
[B,H_q,T,T]
$$

普通实现若显式保存完整 Score 或权重，其元素数也随 $T^2$ 增长。FlashAttention 等精确分块算法可以显著降低这部分中间显存和显存读写，但不会把精确全注意力的算术复杂度从 $O(T^2)$ 改成 $O(T)$。

### 5.2 Decode 的 Attention 计算

使用 KV Cache 时，每一步只有一个新 Query：

~~~text
Q_new:  [B,Hq,1,Dh]
K_all:  [B,Hkv,T,Dh]
scores: [B,Hq,1,T]
~~~

因此单步 Attention 的算术量随当前上下文长度线性增长：

$$
O(BH_qTd_h)=O(BTD)
$$

注意这里的计算头数仍是 $H_q$。GQA/MQA 让多个 Query Head 共享 K/V，主要减少缓存容量和 K/V 读取量，并不把 Query Head 数变成 $H_{kv}$。

若在 Prompt 长度 $T_p$ 后连续生成 $N$ 个 Token，所有 Decode 步的 Attention 长度之和约为：

$$
\sum_{i=0}^{N-1}(T_p+i)
=
NT_p+\frac{N(N-1)}{2}
$$

因此整个续写阶段的 Attention 计算约为：

$$
O\left(BD\left(NT_p+N^2\right)\right)
$$

单步是关于当前长度的线性开销，不代表生成很多 Token 后的总开销仍与生成长度线性。

### 5.3 KV Cache 的容量

单层 K、V 合计字节数：

$$
M_{KV}^{layer}
=
2BTH_{kv}d_hs
$$

$L$ 层合计：

$$
\boxed{
M_{KV}
=
2LBTH_{kv}d_hs
}
$$

每新增一个 Token，理论缓存增量为：

$$
\Delta M_{token}
=
2LBH_{kv}d_hs
$$

这说明 KV Cache 对层数、Batch Size、序列长度、KV Head 数、Head 维度和元素字节数都是线性增长。

该公式只计算有效 Token 的 K、V 张量，不包含分配器预留、对齐、Padding、缓存对象元数据和其他运行时工作区。实际显存增量也会包含模型权重、临时激活与框架开销，不能直接用进程显存减法替代理论 K/V 字节数。

### 5.4 数值例子

设：

~~~text
L = 32
B = 1
T = 4096
Hq = 32
Hkv = 8
Dh = 128
dtype = BF16，s = 2 bytes
~~~

则：

$$
M_{KV}
=
2\times32\times1\times4096\times8\times128\times2
=
536{,}870{,}912\ \text{bytes}
$$

按二进制单位换算：

$$
M_{KV}=512\ \text{MiB}=0.5\ \text{GiB}
$$

每增加一个 Token：

$$
\Delta M_{token}
=
2\times32\times1\times8\times128\times2
=
131{,}072\ \text{bytes}
=
128\ \text{KiB}
$$

若 Batch Size 变成 8，其他条件不变，仅 K/V 理论容量就是 $4\ \text{GiB}$。

### 5.5 长度翻倍时分别发生什么

固定其他量，将长度从 $T$ 变成 $2T$：

| 项目 | 约增加到 | 原因 |
|---|---:|---|
| Prefill 全注意力计算 | 4 倍 | $T^2\rightarrow(2T)^2$ |
| 显式 Attention 矩阵元素数 | 4 倍 | T 轴与 T 轴都翻倍 |
| KV Cache 容量 | 2 倍 | 缓存沿 Token 轴线性增长 |
| 单个 Decode Token 的 Attention | 2 倍 | 一个 Query 要读取两倍历史 K/V |
| 模型权重 | 不变 | 与本次序列长度无关 |

“Attention 显存是二次增长”和“KV Cache 显存是线性增长”可以同时成立，因为二者描述的不是同一种张量。

---

<a id="kv-6-mha-mqa-gqa"></a>

## 6. MHA、GQA、MQA 的结构差异

三者通常保持 Query Head 数 $H_q$ 不变，区别是有多少组独立的 Key/Value Head。

### 6.1 MHA：Multi-Head Attention

MHA 中每个 Query Head 都有自己对应的 Key Head 和 Value Head：

$$
H_{kv}=H_q
$$

例如 $H_q=8$：

~~~text
Q0 -> K0,V0    Q1 -> K1,V1    ...    Q7 -> K7,V7
~~~

形状：

~~~text
Q: [B,8,T,Dh]
K: [B,8,T,Dh]
V: [B,8,T,Dh]
~~~

每个 Query Head 有独立 K/V 表示，表达容量较充分，但缓存最大。

### 6.2 MQA：Multi-Query Attention

MQA 保留多个 Query Head，但所有 Query Head 共享一组 Key 和 Value：

$$
H_{kv}=1
$$

例如 $H_q=8$：

~~~text
Q0 --+
Q1 --+
...   +--> K0,V0
Q7 --+
~~~

形状：

~~~text
Q: [B,8,T,Dh]
K: [B,1,T,Dh]
V: [B,1,T,Dh]
~~~

它最大幅度压缩 K/V，但不同 Query Head 不能拥有完全独立的 K/V 表示。

### 6.3 GQA：Grouped-Query Attention

GQA 介于二者之间：将 Query Head 分组，每组共享一个 KV Head。

$$
1<H_{kv}<H_q
$$

$$
g=\frac{H_q}{H_{kv}}
$$

例如 $H_q=8,H_{kv}=2,g=4$：

~~~text
Q0,Q1,Q2,Q3 -> K0,V0
Q4,Q5,Q6,Q7 -> K1,V1
~~~

形状：

~~~text
Q: [B,8,T,Dh]
K: [B,2,T,Dh]
V: [B,2,T,Dh]
~~~

GQA 用少于 MHA、但多于 MQA 的 K/V Head，在 K/V 表达能力和缓存、带宽成本之间折中。

### 6.4 统一对比

| 结构 | KV Head 数 | 每组 Query Head 数 | K/V 是否共享 | 相对 MHA 的理论 KV Cache |
|---|---:|---:|---|---:|
| MHA | $H_q$ | 1 | 不共享 | $1$ |
| GQA | $H_{kv}$，且 $1<H_{kv}<H_q$ | $H_q/H_{kv}$ | 组内共享 | $H_{kv}/H_q$ |
| MQA | 1 | $H_q$ | 全部共享 | $1/H_q$ |

三者的 Query 形状都是 <code>[B,Hq,T,Dh]</code>，最终 Attention 输出也都要为 $H_q$ 个 Query Head 产生结果。不能把 MQA 理解为“模型只剩一个 Attention Head”。

---

<a id="kv-7-cache-reduction"></a>

## 7. MQA/GQA 为什么减少 KV Cache 占用

缓存公式只包含 $H_{kv}$，不包含 $H_q$：

$$
M_{KV}=2LBTH_{kv}d_hs
$$

若 MHA 的 $H_{kv}=H_q$，则它的缓存为：

$$
M_{MHA}=2LBTH_qd_hs
$$

GQA/MQA 相对 MHA 的比例为：

$$
\frac{M_{GQA/MQA}}{M_{MHA}}
=
\frac{H_{kv}}{H_q}
$$

压缩倍数为：

$$
\frac{M_{MHA}}{M_{GQA/MQA}}
=
\frac{H_q}{H_{kv}}
$$

以前述 $L=32,T=4096,d_h=128,B=1,s=2,H_q=32$ 为例：

| 结构 | $H_{kv}$ | 每组 Query 数 | 理论 KV Cache | 相对 MHA |
|---|---:|---:|---:|---:|
| MHA | 32 | 1 | 2 GiB | 100% |
| GQA | 8 | 4 | 512 MiB | 25% |
| MQA | 1 | 32 | 64 MiB | 3.125% |

减少 KV Head 还会缩小 K/V 投影输出宽度：

$$
W_K,W_V:
D\times(H_{kv}d_h)
$$

因此也减少 K/V 投影参数和部分计算。Decode 往往需要从显存反复读取很长的 K/V；更小的缓存也意味着更少的 K/V 内存流量。这通常是 MQA/GQA 对推理解码很重要的原因。

但需要保持结论边界：

- Query Head 数和 Query 投影没有按相同比例减少；
- 逻辑 Attention 输出仍有 $H_q$ 个头；
- 每个 Query Head 仍需与整段历史进行点积；
- 端到端速度还受 Kernel、硬件、Batch、长度和其他层影响；
- 更强的 K/V 共享可能带来表达能力或效果取舍，不能仅凭缓存公式判断模型质量。

---

<a id="kv-8-effective-context"></a>

## 8. 长上下文不等于模型能有效利用全部信息

### 8.1 三个概念必须分开

~~~text
可接收长度：系统允许输入多少 Token，不报长度错误或 OOM。
可计算长度：位置编码、Attention、Cache 和运行时能够处理到多长。
有效利用长度：模型能否稳定检索、组合并正确使用远距离信息。
~~~

模型声明的 Context Window 主要给出前两类能力的上限，不自动证明第三类能力。

### 8.2 为什么信息在窗口内仍可能没被有效使用

可能原因包括：

- 训练阶段很少见到如此长的序列或相应位置分布；
- RoPE 等位置方法在训练长度外需要外推，位置关系可能退化；
- 有关信息被大量无关 Token 稀释；
- 模型对信息所在位置敏感，例如中部信息利用较弱；
- 任务需要跨多处证据组合，而不只是精确复制一个片段；
- 输入被 Tokenizer、聊天模板或上游程序截断；
- 生成长度预留、系统提示和对话历史共同占用总窗口。

因此“某条信息被放进 Context”不等于“模型一定记住、检索到并正确推理”。

### 8.3 正确评测长上下文

至少改变以下变量：

~~~text
总上下文长度
关键信息位置：开头 / 中间 / 结尾
无关干扰信息数量
任务类型：精确检索 / 多证据组合 / 推理
答案是否能由外部规则核验
~~~

只测试一个 Prompt、一个位置或一个“针在草堆中”样例，不能证明模型能普遍利用全部 Context。应记录准确率随长度和信息位置的变化，并把“成功运行”与“回答正确”分开。

### 8.4 KV Cache 与模型能力的边界

KV Cache 是计算复用机制：

$$
\text{相同历史 K/V，避免重复投影和重复历史前向}
$$

它不会：

- 扩大模型训练时学到的位置泛化能力；
- 自动选择最重要的上下文；
- 修复模型的检索或多步推理错误；
- 让超过模型或运行时上限的 Token 自动可用。

更大的缓存容量解决“能否保存更多历史状态”，不等于解决“能否有效使用更多历史信息”。

---

<a id="kv-9-practice"></a>

## 9. 完整实践任务

下面使用 Hugging Face Transformers 观察真实 Decoder-only 模型。示例默认 Batch Size 为 1、无 Padding，并使用 Greedy Decoding，以便只比较 Cache 计算路径，不混入随机采样差异。

依赖：

~~~bash
pip install torch transformers
~~~

可选择任意能放入当前设备的开源 Decoder-only Causal LM：

~~~python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    LOAD_DTYPE = torch.bfloat16
else:
    LOAD_DTYPE = torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=LOAD_DTYPE,
).to(DEVICE)
model.eval()

prompt = "KV Cache 的主要作用是"
encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded.input_ids.to(DEVICE)
attention_mask = encoded.attention_mask.to(DEVICE)

print("model:", MODEL_ID)
print("device:", DEVICE)
print("input_ids:", tuple(input_ids.shape))
~~~

若设备不支持 BF16，将 <code>LOAD_DTYPE</code> 改为 <code>torch.float16</code> 或 <code>torch.float32</code>。实践报告必须记录模型 ID、Transformers/PyTorch 版本、设备、Dtype、Prompt Token 数和生成 Token 数。

### 9.1 Cache 兼容辅助函数

不同 Transformers 版本可能返回 Legacy Tuple 或 Cache 类。以下函数只负责把它转换成便于观察的每层 K、V 列表：

~~~python
def cache_to_layers(past_key_values):
    if past_key_values is None:
        return []

    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    layers = []
    for layer_cache in past_key_values:
        if len(layer_cache) < 2:
            raise ValueError("每层 Cache 至少应包含 K 和 V")
        layers.append((layer_cache[0], layer_cache[1]))
    return layers


def cache_nbytes(past_key_values):
    total = 0
    for key, value in cache_to_layers(past_key_values):
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


def sync_device():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
~~~

这段函数用于教学观察，不代表推理框架内部一定以 Legacy Tuple 物理存储缓存。

---

<a id="kv-practice-1-observe-shapes"></a>

### 实践 1：观察各层 Hidden State 和 KV Cache 形状

先执行 Prefill：

~~~python
with torch.inference_mode():
    prefill_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

hidden_states = prefill_outputs.hidden_states
cache_layers = cache_to_layers(prefill_outputs.past_key_values)

print("hidden state count:", len(hidden_states))
for index, hidden in enumerate(hidden_states):
    print(f"hidden[{index:02d}] = {tuple(hidden.shape)}")

print("cache layer count:", len(cache_layers))
for index, (key, value) in enumerate(cache_layers):
    print(
        f"layer={index:02d} "
        f"K={tuple(key.shape)} "
        f"V={tuple(value.shape)} "
        f"dtype={key.dtype}"
    )

print("actual cache bytes:", cache_nbytes(prefill_outputs.past_key_values))
~~~

常见预期：

~~~text
len(hidden_states) = L + 1
每个 Hidden State: [B,Tp,D]
len(cache_layers) = L
每层 K/V: [B,Hkv,Tp,Dh]
~~~

<code>hidden_states[0]</code> 通常是进入第一层前的表示，后续元素对应每层输出；具体命名应以模型实现为准。<code>output_hidden_states=True</code> 是为了观察，正常生成无需保留所有 Hidden State。

再输入一个新 Token，观察 Decode 后 Cache 长度增加 1：

~~~python
first_next_id = prefill_outputs.logits[:, -1, :].argmax(
    dim=-1,
    keepdim=True,
)

extended_mask = torch.cat(
    [
        attention_mask,
        torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        ),
    ],
    dim=1,
)

with torch.inference_mode():
    decode_outputs = model(
        input_ids=first_next_id,
        attention_mask=extended_mask,
        past_key_values=prefill_outputs.past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

for index, hidden in enumerate(decode_outputs.hidden_states):
    print(f"decode hidden[{index:02d}] = {tuple(hidden.shape)}")

for index, (key, value) in enumerate(
    cache_to_layers(decode_outputs.past_key_values)
):
    print(f"decode layer={index:02d} K={tuple(key.shape)} V={tuple(value.shape)}")
~~~

常见预期：

~~~text
Decode Hidden State 的 Token 轴：1
Decode 后 Cache 的 Token 轴：Tp + 1
~~~

如果模型要求显式 <code>cache_position</code> 或特殊 Position ID，应按该模型当前版本的 Forward API 传入；不要用错误位置强行凑出正确形状。

验收记录：

| 项目 | 观测值 |
|---|---|
| $L$ |  |
| $D$ |  |
| $H_q$ |  |
| $H_{kv}$ |  |
| $d_h$ |  |
| Prompt 长度 |  |
| Prefill 第 0/最后 Hidden State 形状 |  |
| Prefill 第 0/最后层 K、V 形状 |  |
| 一次 Decode 后 K、V 形状 |  |
| 实际 K/V 元素字节数 |  |

---

<a id="kv-practice-2-cache-on-off"></a>

### 实践 2：对比启用与关闭 KV Cache 的逐 Token 解码

关闭 Cache 时，每步重新输入完整上下文：

~~~python
@torch.inference_mode()
def greedy_without_cache(model, input_ids, attention_mask, max_new_tokens):
    generated = input_ids.clone()
    mask = attention_mask.clone()

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        mask = torch.cat(
            [
                mask,
                torch.ones(
                    (mask.size(0), 1),
                    dtype=mask.dtype,
                    device=mask.device,
                ),
            ],
            dim=1,
        )

    return generated
~~~

开启 Cache 时，第一步执行完整 Prompt Prefill，后续每步只输入上一步新选出的 Token：

~~~python
@torch.inference_mode()
def greedy_with_cache(model, input_ids, attention_mask, max_new_tokens):
    generated = input_ids.clone()
    mask = attention_mask.clone()
    past_key_values = None
    current_input = input_ids

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_input,
            attention_mask=mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)

        past_key_values = outputs.past_key_values
        current_input = next_id
        mask = torch.cat(
            [
                mask,
                torch.ones(
                    (mask.size(0), 1),
                    dtype=mask.dtype,
                    device=mask.device,
                ),
            ],
            dim=1,
        )

    return generated
~~~

对齐行为并计时：

~~~python
MAX_NEW_TOKENS = 64

# 预热，避免把首次 Kernel 初始化等成本只算入某一组。
_ = greedy_without_cache(model, input_ids, attention_mask, 2)
_ = greedy_with_cache(model, input_ids, attention_mask, 2)

sync_device()
start = time.perf_counter()
tokens_without_cache = greedy_without_cache(
    model,
    input_ids,
    attention_mask,
    MAX_NEW_TOKENS,
)
sync_device()
time_without_cache = time.perf_counter() - start

sync_device()
start = time.perf_counter()
tokens_with_cache = greedy_with_cache(
    model,
    input_ids,
    attention_mask,
    MAX_NEW_TOKENS,
)
sync_device()
time_with_cache = time.perf_counter() - start

print("same token ids:", torch.equal(tokens_without_cache, tokens_with_cache))
print("without cache:", time_without_cache)
print("with cache:", time_with_cache)
print("speedup:", time_without_cache / time_with_cache)
print(tokenizer.decode(tokens_with_cache[0], skip_special_tokens=False))
~~~

公平比较要求：

~~~text
同一模型与 Dtype
model.eval() 和 torch.inference_mode()
同一 Prompt 和 Token 数
同一 Greedy 规则
两组都预热
GPU 计时前后同步
重复多次并报告中位数
不把模型加载、Tokenizer 和文本解码计入前向耗时
~~~

短 Prompt、只生成一两个 Token 时，Cache 管理开销可能掩盖收益。应同时测试多个 Prompt 长度和生成长度，不能从一个短样例推导普遍加速比。

建议记录：

| Prompt Token 数 | 生成 Token 数 | 无 Cache 总时间 | 有 Cache 总时间 | 加速比 | Token ID 是否一致 |
|---:|---:|---:|---:|---:|---|
| 32 | 64 |  |  |  |  |
| 128 | 64 |  |  |  |  |
| 512 | 64 |  |  |  |  |

若要解释首 Token 延迟与稳态 Decode 延迟，应进一步分开记录 Prefill 时间和每个 Decode Token 的时间，不能只看一个总时间。

---

<a id="kv-practice-3-estimate-size"></a>

### 实践 3：估算并核对 KV Cache 大小

从模型配置读取参数：

~~~python
config = model.config

L = config.num_hidden_layers
Hq = config.num_attention_heads
Hkv = getattr(config, "num_key_value_heads", Hq)
Dh = getattr(config, "head_dim", config.hidden_size // Hq)
B = input_ids.size(0)
T = input_ids.size(1)
element_size = next(model.parameters()).element_size()

estimated_bytes = 2 * L * B * T * Hkv * Dh * element_size

with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

actual_bytes = cache_nbytes(outputs.past_key_values)

print({
    "layers": L,
    "query_heads": Hq,
    "kv_heads": Hkv,
    "head_dim": Dh,
    "batch_size": B,
    "sequence_length": T,
    "element_size": element_size,
    "estimated_bytes": estimated_bytes,
    "actual_tensor_bytes": actual_bytes,
    "equal": estimated_bytes == actual_bytes,
    "estimated_MiB": estimated_bytes / 2**20,
})
~~~

这里用模型参数的 <code>element_size()</code> 作为缓存 Dtype 的初始估计。严格核对时应直接检查实际 Key/Value 的 <code>dtype</code> 和 <code>element_size()</code>；某些系统会让权重 Dtype 与缓存 Dtype 不同。

再验证长度线性关系：

~~~python
def estimate_kv_mib(L, B, T, Hkv, Dh, element_size):
    byte_count = 2 * L * B * T * Hkv * Dh * element_size
    return byte_count / 2**20


for length in [128, 512, 2048, 8192, 32768]:
    mib = estimate_kv_mib(L, B, length, Hkv, Dh, element_size)
    print(f"T={length:>6}: {mib:>10.2f} MiB")
~~~

验收要求：手算值、函数估算值、实际 K/V 张量字节数三者应一致；若不一致，逐项核对层数、KV Head 数、Head 维度、有效 Token 数和实际 Cache Dtype。

---

<a id="kv-practice-4-compare-attention-types"></a>

### 实践 4：对比 MHA、GQA、MQA 的理论缓存量

固定 $L,B,T,H_q,d_h,s$，只改变 $H_{kv}$：

~~~python
def compare_attention_cache(
    num_layers,
    batch_size,
    sequence_length,
    query_heads,
    head_dim,
    element_size,
    gqa_kv_heads,
):
    configs = {
        "MHA": query_heads,
        "GQA": gqa_kv_heads,
        "MQA": 1,
    }

    mha_bytes = None
    rows = []
    for name, kv_heads in configs.items():
        if query_heads % kv_heads != 0:
            raise ValueError("query_heads 必须能被 kv_heads 整除")

        byte_count = (
            2
            * num_layers
            * batch_size
            * sequence_length
            * kv_heads
            * head_dim
            * element_size
        )
        if name == "MHA":
            mha_bytes = byte_count

        rows.append({
            "type": name,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "queries_per_kv": query_heads // kv_heads,
            "MiB": byte_count / 2**20,
            "ratio_to_mha": byte_count / mha_bytes,
        })

    return rows


rows = compare_attention_cache(
    num_layers=32,
    batch_size=1,
    sequence_length=4096,
    query_heads=32,
    head_dim=128,
    element_size=2,
    gqa_kv_heads=8,
)

for row in rows:
    print(row)
~~~

预期输出应满足：

~~~text
MHA: 2048 MiB，ratio = 1.0
GQA:  512 MiB，ratio = 0.25
MQA:   64 MiB，ratio = 0.03125
~~~

还应验证：

$$
\text{ratio\_to\_mha}
=
\frac{H_{kv}}{H_q}
$$

只把 $H_{kv}$ 从 $H_q$ 改成 1 是理论形状和容量对比，不等于把一个已经训练好的 MHA 模型无损转换成 MQA。结构改变涉及投影参数和训练过程。

---

<a id="kv-10-common-errors"></a>

## 10. 常见错误

### 10.1 把 Cache 长度理解成“已经输出到屏幕的 Token 数”

模型本次输出的 Logits 用于选择下一个 Token；这个刚选出的 Token 通常要到下一次模型调用时才写入 Cache。因此必须明确观察点位于调用前还是调用后。

### 10.2 使用 Cache 时仍传入完整历史输入

已有完整历史 Cache 后，普通 Decode 每步只输入未缓存的新 Token。若把完整历史 Token 与相同历史 Cache 一起传入，可能重复位置、Mask 或长度不匹配。某些高级接口允许不同输入形式，但必须遵守该接口的明确契约。

### 10.3 只传新 Token，却不传完整有效 Attention Mask

许多 Hugging Face 模型在使用 Past KV 时要求 Attention Mask 覆盖“历史缓存 + 当前输入”的总长度。输入 ID 可以只有 1 个 Token，Mask 长度不一定也是 1。

### 10.4 用 Query Head 数估算 GQA/MQA Cache

缓存公式使用 $H_{kv}$：

$$
M_{KV}=2LBTH_{kv}d_hs
$$

使用 $H_q$ 会把 GQA/MQA 的缓存高估 $H_q/H_{kv}$ 倍。

### 10.5 忘记 K 和 V 是两个张量

若只计算一个张量，会漏掉公式最前面的系数 2。

### 10.6 混淆十进制 GB 与二进制 GiB

$$
1\ \text{GB}=10^9\ \text{bytes}
$$

$$
1\ \text{GiB}=2^{30}\ \text{bytes}
$$

报告数值时应标明单位。

### 10.7 用进程显存差值直接验证公式

进程显存还包含模型权重、临时激活、Kernel Workspace、内存池保留和碎片。理论公式应先与 Cache 中 K、V 张量的 <code>numel() * element_size()</code> 对照。

### 10.8 认为 KV Cache 将所有复杂度都变成常数

Cache 避免历史位置的重复前向，但新 Query 仍要读取长度为 $T$ 的历史 K/V，单步 Decode Attention 仍是 $O(T)$。

### 10.9 认为更大的 Context Window 必然得到更好的答案

Context 上限表示可容纳范围，不是有效利用保证。必须用位置、长度、干扰量和任务类型受控的实验评估。

### 10.10 把 GQA/MQA 等同于少做相同比例的全部 Attention 计算

GQA/MQA 缩小 K/V Head 数，但 Query Head 数通常不变。最直接收益是 K/V Cache、K/V 投影和 Decode 内存流量下降，不是整个模型 FLOPs 按缓存比例缩小。

---

<a id="kv-11-checklist"></a>

## 11. 本章验收

- [ ] 能说明 Prefill 一次处理整个 Prompt，Decode 每次处理新 Token 并复用历史 Cache。
- [ ] 能解释 Prefill 后 Cache 已包含 Prompt，但尚未包含刚由最后位置 Logits 选出的首个生成 Token。
- [ ] 能说明每个 Transformer 层分别缓存历史 Token 的 Key 和 Value。
- [ ] 能解释标准 KV Cache 为什么不需要保存历史 Query、Attention 权重和全部 Hidden State。
- [ ] 能用 Causal Mask 解释历史 Token 的 K、V 为什么可以在右侧追加 Token 后复用。
- [ ] 能从 $X:[B,T,D]$ 和投影矩阵推导 $Q:[B,H_q,T,d_h]$、$K,V:[B,H_{kv},T,d_h]$。
- [ ] 能推导单层 K、V 元素数 $2BTH_{kv}d_h$。
- [ ] 能推导完整模型缓存公式 $2LBTH_{kv}d_hs$ bytes。
- [ ] 能区分 Prefill Attention 的 $O(T^2)$、有 Cache 的单步 Decode Attention 的 $O(T)$ 和 KV Cache 容量的 $O(T)$。
- [ ] 能说明 FlashAttention 降低中间显存和显存读写，但不改变精确全注意力的 $O(T^2)$ 算术复杂度。
- [ ] 能分别画出 MHA、GQA、MQA 的 Query Head 到 KV Head 映射。
- [ ] 能用 $H_{kv}/H_q$ 计算 GQA/MQA 相对 MHA 的缓存比例。
- [ ] 能解释 MQA 仍有多个 Query Head，不能称为只有一个 Attention Head。
- [ ] 能说明 KV Cache 以空间换时间，但不消除自回归串行依赖。
- [ ] 能区分可接收长度、可计算长度和有效利用长度。
- [ ] 能独立完成四项实践，并记录模型配置、张量形状、理论字节数、实际字节数和计时条件。

[返回学习列表](学习列表.md#learning-list-kv-cache-long-context)
