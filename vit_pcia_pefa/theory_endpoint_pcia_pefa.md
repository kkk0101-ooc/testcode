# Endpoint–PCIA–PEFA 理论链、等价边界与表征条件

- 状态：`PROVED_UNDER_FROZEN_C2_GRAPH_SCOPE`
- 日期：2026-09-21（PRC）
- 方法定位：`COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE`
- 适用范围：固定有限计算图、分离的真实消费端口、合法完整切面、二阶光滑邻域和分段 $C^1$ 输入路径

本文给出步骤 3 的理论结果。第 1–7 节只陈述定义、命题、证明和反例；第 8 节单列步骤 2 的数值观察。所有恒等式都是精确等式，数值容差只用于实现验证。

## 1. 设定与充分条件

### 1.1 冻结计算对象

令 $U\subset\mathbb R^d$ 为开集，$F:U\to\mathbb R$ 为标量目标，$b,x\in U$。路径 $\gamma:[0,1]\to U$ 分段 $C^1$，满足 $\gamma(0)=b$、$\gamma(1)=x$。除输入根 $u$ 外，模型参数、buffer、目标索引和随机状态均固定。

以下条件是本文全部核心定理的一组充分条件。

1. **光滑性。** 冻结图中从根到切面端口、再从端口到目标的组成映射在包含 $\gamma([0,1])$ 的开邻域 $V\subset U$ 上为 $C^2$。等价地，本文直接要求 $F\in C^2(V)$ 且每个端口场 $D_e\in C^1(V)$。因此所需端口导数存在，$F$ 的混合偏导可交换。
2. **图冻结。** $V$ 上的执行拓扑、consumer use、input slot、invocation、tensor shape 和目标不变。若控制流改变图，只能在一个固定图分支内分别应用本文结论。
3. **端口分离。** 端口 $e$ 标识一次 producer value 到一个确定 consumer、input slot 和 invocation 的 use。fan-out 的不同消费者使用各自的下游余切；不得先在 producer 上合并余切再复制。
4. **合法完整切面。** 切面 $C$ 与冻结图中每条从所有变化根到 $F$ 的有效有向路径恰好相交一次，而且该性质在 $V$ 上保持。
5. **路径条件。** $\gamma$ 的像包含在 $V$ 中，$\dot\gamma$ 分段连续且可积。路径可以弯曲或自交；路径选择本身不被冻结为不变量。
6. **根变量完备。** 所有随路径变化的独立根都包含在 $u$ 中。若另一个根随 $t$ 变化却未纳入 $u$，链式法则会缺项。

这些条件有意排除跨越 ReLU kink 的路径和沿路径改变端口集合的动态图。第 7 节给出范围收缩方式。

### 1.2 真实使用端口与切面场

对端口 $e$，令 $z_e(u)$ 表示送入该 consumer use 的值，$\bar z_e(u)$ 表示只沿该 use 的下游子图回传到 $z_e$ 的余切。定义根坐标上的端口场

\[
D_e(u)=J_{z_e,u}(u)^{\mathsf T}\bar z_e(u),
\qquad
D_{e,i}(u)=[D_e(u)]_i.
\]

同一个 producer 的两个 consumer use 可以具有相同的 $J_{z_e,u}$，但必须使用不同的 $\bar z_e$。合法切面把反向链式法则中的所有根—目标微分路径无遗漏、无重复地分组，因此

\[
\sum_{e\in C}D_e(u)=\nabla F(u),\qquad u\in V.
\tag{1}
\]

式 (1) 是切面结论的图论前提，不由数值闭合反推。

### 1.3 Endpoint、PCIA、PEFA 与路由曲率

记 $\Delta=x-b$。参考点相对 Endpoint 及其端口分量为

\[
E_i^b(x)=\Delta_i\,\partial_iF(x),
\qquad
Q_{e,i}^b(u)=(u_i-b_i)D_{e,i}(u).
\tag{2}
\]

一般路径 IG、PCIA 一阶项、PEFA 形成张量和累计二阶项分别定义为

\[
\operatorname{IG}_i^\gamma(F)
=\int_0^1\dot\gamma_i(t)\partial_iF(\gamma(t))\,dt,
\tag{3}
\]

\[
A_{e,i}^\gamma
=\int_0^1\dot\gamma_i(t)D_{e,i}(\gamma(t))\,dt,
\tag{4}
\]

\[
T_{e,i\leftarrow j}^\gamma
=\int_0^1\dot\gamma_j(t)
\partial_jQ_{e,i}^b(\gamma(t))\,dt,
\tag{5}
\]

\[
R_{e,i\leftarrow j}^\gamma
=\int_0^1(\gamma_i(t)-b_i)\dot\gamma_j(t)
\partial_jD_{e,i}(\gamma(t))\,dt.
\tag{6}
\]

端口路由曲率定义为端口场 Jacobian 的反对称部分

\[
\Omega_{e,ij}(u)
=\partial_iD_{e,j}(u)-\partial_jD_{e,i}(u).
\tag{7}
\]

$R$ 是沿指定输入路径累计的加权二阶变化项；$\Omega$ 是某一点处端口场是否为局部保守场的反对称诊断。这里的“曲率”不指 Riemann 曲率。

## 2. 基础引理

### 引理 1：合法切面的微分闭合

在第 1.1 节条件下，式 (1) 对 $V$ 中每个 $u$ 成立。进而

\[
\sum_{e\in C}Q_{e,i}^b(u)
=(u_i-b_i)\partial_iF(u).
\tag{8}
\]

**证明。** 标量目标对根的总微分可展开为冻结 DAG 上所有根—目标有向微分路径的乘积之和。合法切面使每条路径落入唯一端口组；对这些组求和恢复完整反向链式法则，即式 (1)。式 (8) 由两边同乘 $u_i-b_i$ 得到。$\square$

### 引理 2：端口 Endpoint 分量的路径微分

对任意 $e,i$，复合函数 $t\mapsto Q_{e,i}^b(\gamma(t))$ 在每个光滑分段上满足

\[
\frac{d}{dt}Q_{e,i}^b(\gamma(t))
=\sum_j\dot\gamma_j(t)\partial_jQ_{e,i}^b(\gamma(t)).
\tag{9}
\]

**证明。** $Q_{e,i}^b$ 为 $C^1$，对其与分段 $C^1$ 路径的复合应用链式法则。有限个分段相加后内部端点相消。$\square$

### 引理 3：乘积导数分解

对任意 $e,i,j$，

\[
\partial_jQ_{e,i}^b(u)
=\delta_{ij}D_{e,i}(u)
 +(u_i-b_i)\partial_jD_{e,i}(u).
\tag{10}
\]

**证明。** 对式 (2) 使用普通乘积法则，且 $\partial_j(u_i-b_i)=\delta_{ij}$。$\square$

## 3. 闭合与差分定理

### 定理 1：逐端口、逐 Endpoint 坐标的 PEFA 闭合

每个端口 $e$ 和 Endpoint 坐标 $i$ 均满足

\[
\boxed{
\sum_jT_{e,i\leftarrow j}^\gamma
=Q_{e,i}^b(x)
}.
\tag{11}
\]

若 $C$ 为合法完整切面，则每个 $i$ 进一步满足

\[
\boxed{
\sum_{e\in C}\sum_jT_{e,i\leftarrow j}^\gamma
=E_i^b(x)
}.
\tag{12}
\]

**证明。** 对式 (9) 从 0 到 1 积分。左边成为式 (5) 对 $j$ 的和，右边为
$Q_{e,i}^b(x)-Q_{e,i}^b(b)$。因 $Q_{e,i}^b(b)=0$，得到式 (11)。再对合法切面求和并在式 (8) 中取 $u=x$，得到式 (12)。$\square$

逐端口闭合不要求端口集合本身构成合法切面；从端口闭合提升到完整 Endpoint 闭合才需要式 (1)。

### 定理 2：直接项与累计二阶项分解

每个 $e,i,j$ 均满足

\[
\boxed{
T_{e,i\leftarrow j}^\gamma
=\delta_{ij}A_{e,i}^\gamma+R_{e,i\leftarrow j}^\gamma
}.
\tag{13}
\]

**证明。** 将式 (10) 代入式 (5)。第一项仅在 $i=j$ 时保留，并由式 (4) 等于 $A_{e,i}^\gamma$；第二项正是式 (6)。$\square$

### 定理 3：Endpoint–IG 逐坐标差分恒等式

对合法完整切面 $C$，每个输入坐标 $i$ 满足

\[
\boxed{
E_i^b(x)-\operatorname{IG}_i^\gamma(F)
=\sum_{e\in C}\sum_jR_{e,i\leftarrow j}^\gamma
}.
\tag{14}
\]

**证明。** 对式 (13) 先求和 $j$、再求和 $e$。由定理 1，左边为 $E_i^b(x)$。直接项为

\[
\sum_eA_{e,i}^\gamma
=\int_0^1\dot\gamma_i(t)
\sum_eD_{e,i}(\gamma(t))dt
=\operatorname{IG}_i^\gamma(F),
\]

其中最后一步使用式 (1)。移项即得式 (14)。$\square$

在完整切面上，$R$ 的端口和可直接写成加权 Hessian 路径积分：

\[
\sum_eR_{e,i\leftarrow j}^\gamma
=\int_0^1(\gamma_i-b_i)\dot\gamma_j
\partial_{ji}^2F(\gamma(t))dt.
\tag{15}
\]

直线路径 $\gamma(t)=b+t\Delta$ 下，式 (15) 化为

\[
\sum_eR_{e,i\leftarrow j}
=\Delta_i\Delta_j\int_0^1t\,\partial_{ji}^2F(b+t\Delta)dt.
\tag{16}
\]

式 (14) 的候选增量限定为式 (15) 在真实端口上的分配；二阶积分运算属于既有方法族。

### 定理 4：完整切面的路由曲率零和

对合法完整切面 $C$，任意 $u\in V$ 与任意 $i,j$ 满足

\[
\boxed{
\sum_{e\in C}\Omega_{e,ij}(u)=0
}.
\tag{17}
\]

**证明。** 因式 (1) 在开邻域 $V$ 上成立，可以逐坐标求导：

\[
\sum_e\Omega_{e,ij}
=\partial_i\sum_eD_{e,j}-\partial_j\sum_eD_{e,i}
=\partial_{ij}^2F-\partial_{ji}^2F=0.
\]

最后一步使用 $F\in C^2(V)$ 的混合偏导对称性。$\square$

仅在路径采样点上偶然满足 $\sum_eD_e=\nabla F$ 不足以推出式 (17)；证明需要邻域内的合法切面恒等式。

## 4. 等价边界

### 4.1 PCIA 与 conductance

对标量隐藏 use $z_e=y$，

\[
D_{e,i}=\frac{\partial y}{\partial u_i}\frac{\partial F}{\partial y},
\qquad
A_{e,i}^\gamma
=\int_0^1\dot\gamma_i
\frac{\partial F}{\partial y}
\frac{\partial y}{\partial u_i}dt.
\]

这就是输入分辨 conductance。向量端口对内部坐标收缩后仍是同一链式法则。因此 PCIA 的贡献范围是 conductance/path-IG 的真实 use-port 与合法切面系统化。

### 4.2 PEFA $T$ 与普通元归因

固定 $e,i,b$，令标量解释对象 $G_{e,i}(u)=Q_{e,i}^b(u)$。式 (5) 立即成为

\[
T_{e,i\leftarrow j}^\gamma
=\operatorname{IG}_j^\gamma(G_{e,i}).
\tag{18}
\]

所以 $T$ 在以下条件下退化为普通元归因：基础解释被选为端口 Endpoint 分量 $Q_{e,i}^b$，元归因与基础解释使用同一根坐标、参考点和输入路径，并且端口身份在路径邻域内固定。若先对端口求和，式 (18) 相应退化为对 $E_i^b$ 做普通 path IG。$T$ 的操作级新颖性据此否定。

### 4.3 与 Integrated Hessians 的边界

完整切面上的 $T$ 与 Integrated Hessians 都包含二阶路径项，但二者解释对象和守恒终点不同：前者解释 $E_i^b$ 并逐 $i$ 守恒到 Endpoint；后者递归解释 $\operatorname{IG}_i(F)$ 并最终守恒到 $F(x)-F(b)$。

取 $F(u_1,u_2)=u_1u_2$、$b=0$、$x=(a,c)$。直线路径上，完整切面 PEFA 的四个输入级元素均为 $ac/2$，每一行求和为 Endpoint 分量 $ac$。按递归 IG 定义，Integrated Hessians 的四个元素均为 $ac/4$，每一行求和为基础 IG 分量 $ac/2$。当 $ac\ne0$ 时两者数值不同。该反例的结论限于具体量和守恒终点；二阶运算的先例归属沿用步骤 1 审计。

### 4.4 与 EAP-IG 和干预归因的边界

EAP-IG 沿 clean–corrupted 内部激活路径积分边梯度，用于近似 patch metric。本文的 $A/T/R$ 沿输入根路径积分，终点分别是 path IG、端口 Endpoint 与 Endpoint–IG 差。两者只有在把路径空间、标量目标、端口变量和终点守恒人为设成同一个对象时才可能重合；通常定义下不等价。

## 5. 表征条件与不变性

| 变化 | 可证明结论 | 不保留的对象或必要条件 |
|---|---|---|
| 路径重参数化 | $A,T,R$ 在保向、分段 $C^1$ 的双射重参数化下不变 | 反向定向会改变积分符号和端点角色 |
| 路径选择 | $\sum_jT_{e,i\leftarrow j}=Q_{e,i}^b(x)$ 与 Endpoint 终点不依赖具体路径 | $A$、各个 $T_{e,i\leftarrow j}$、$R$ 和 IG 通常依赖路径 |
| 参考点 | 无一般参考不变性 | $E,Q,T,R$ 显式依赖 $b$；路径也随端点改变 |
| 输入坐标变换 | 对逐坐标仿射缩放和平移，$E,Q,A,T,R$ 保持同索引数值；坐标置换只重标索引 | 一般坐标混合会重新分配 Endpoint/IG/PEFA；$\Omega$ 在缩放下按二阶协变因子变换 |
| 端口内可逆重参数化 | 若只以可逆 $C^2$ 坐标替换同一个 use，并以精确逆映射改写 consumer，则根坐标上的 $D,Q,A,T,R,\Omega$ 不变 | 改变端口边界、丢维、近似逆或混合不同 consumer 不受保证 |
| 并行细化 | 若细化满足 $D_e=\sum_kD_{e_k}$ 于整个邻域，则各细化项之和恢复原 $Q,A,T,R,\Omega$ | 单个细化项没有不变性；数值上碰巧相加不足以证明结构合法 |
| 跨图重写 | 若新旧图实现同一 $F$，总 Endpoint、IG 和完整切面总和保持 | 端口级 $D,Q,T,R,\Omega$ 一般不保持，也不自动存在端口双射 |

### 5.1 路径重参数化

令 $\phi:[0,1]\to[0,1]$ 为保向、分段 $C^1$ 双射，$\tilde\gamma(s)=\gamma(\phi(s))$。对式 (4)–(6) 使用 $\dot{\tilde\gamma}(s)=\dot\gamma(\phi(s))\phi'(s)$ 和一维换元公式，可得

\[
A^{\tilde\gamma}=A^\gamma,\qquad
T^{\tilde\gamma}=T^\gamma,\qquad
R^{\tilde\gamma}=R^\gamma.
\tag{19}
\]

这只消除沿同一有向几何路径的速度选择，不消除不同几何路径之间的差异。

### 5.2 路径选择与参考点

定理 1 表明，$T$ 对来源坐标 $j$ 求和后的端口终值仅由端点决定；式 (3)–(6) 的各坐标分配通常仍依赖整条路径。以 $F(u_1,u_2)=u_1u_2$、$b=(0,0)$、$x=(1,1)$ 为例，直线路径给出 $\operatorname{IG}=(1/2,1/2)$；先移动 $u_1$、再移动 $u_2$ 的折线路径给出 $(0,1)$。

参考点也属于解释定义。对 $F(u)=u^2$、$x=1$，$b=0$ 时 $E^b=2$，$b=1/2$ 时 $E^b=1$。因此不同 reference 信息条件必须分组报告。

### 5.3 输入坐标变换

先取逐坐标仿射变换 $v_k=a_ku_k+c_k$，其中 $a_k\ne0$，并令 $\beta_k=a_kb_k+c_k$。链式法则给出

\[
D_{e,k}^{(v)}=a_k^{-1}D_{e,k}^{(u)},\qquad
Q_{e,k}^{\beta,(v)}=Q_{e,k}^{b,(u)}.
\]

路径速度与坐标导数中的比例因子相消，所以 $E,Q,A,T,R$ 的同索引数值保持；坐标置换只同步重标 $i,j$。曲率按

\[
\Omega_{e,k\ell}^{(v)}=(a_ka_\ell)^{-1}
\Omega_{e,k\ell}^{(u)}
\tag{20}
\]

变换，故其数值尺度不保持，但“为零”和完整切面零和保持。

一般可逆坐标混合没有逐坐标不变性。例如 $F(u_1,u_2)=u_1$，$b=0$，令 $v_1=u_1+u_2$、$v_2=u_1-u_2$。在 $x=(1,0)$ 处，原坐标 Endpoint 为 $(1,0)$，新坐标 Endpoint 为 $(1/2,1/2)$。因此主合同冻结 normalized RGB 坐标，不能把旋转、白化或跨通道混合后的分量与原分量直接等同。

### 5.4 端口内可逆重参数化

令 $z'_e=h(z_e)$，其中 $h$ 在端口值邻域内为 $C^2$ 微分同胚，并将该 consumer 精确改写为 $c'=c\circ h^{-1}$。则

\[
J_{z'_e,u}^{\mathsf T}\nabla_{z'_e}F
=J_{z_e,u}^{\mathsf T}J_h^{\mathsf T}J_h^{-\mathsf T}
\nabla_{z_e}F
=D_e.
\tag{21}
\]

等式在邻域内逐点成立，因此对根坐标求导后 $\partial_jD_{e,i}$ 也保持，进而 $Q,A,T,R,\Omega$ 全部保持。该结论是同一个 use 内的坐标不变性，不是任意网络重写的实现不变性。

### 5.5 并行细化

若一个端口被细化为 $e_1,\ldots,e_m$，且在 $V$ 上具有可微的点态加法关系

\[
D_e(u)=\sum_{k=1}^mD_{e_k}(u),
\tag{22}
\]

则定义的线性性和对式 (22) 求导给出

\[
Q_e=\sum_kQ_{e_k},\quad
A_e=\sum_kA_{e_k},\quad
T_e=\sum_kT_{e_k},\quad
R_e=\sum_kR_{e_k},\quad
\Omega_e=\sum_k\Omega_{e_k}.
\tag{23}
\]

互斥且完备的 head、channel 或 token slice 可按此条件审计。式 (23) 只保证细化后的和，不为单个细化项提供规范唯一性。

### 5.6 跨图重写反例

同一函数 $F(u_1,u_2)=u_1u_2$ 可以用一个完整输入端口表示，此时 $D=(u_2,u_1)$ 且 $\Omega=0$；也可以按乘法算子的两个 consumer slot 表示，此时

\[
D_1=(u_2,0),\qquad D_2=(0,u_1),
\]

两个端口分别具有相反的非零反对称项，求和后才为零。输出函数、Endpoint 和完整切面总量相同，端口级对象不同。跨图算子融合、公共子表达式消除、fan-out 改写或模块复用变化因此必须建立新的端口 registry，不能继承旧端口结论。

## 6. 退化情形与反例

### 命题 5：仿射图退化

若冻结 DAG 的每条端口场 $D_e$ 在 $V$ 上为常量，特别是所有节点均为仿射映射的图，则

\[
R_{e,i\leftarrow j}=0,\qquad
\Omega_{e,ij}=0,\qquad
T_{e,i\leftarrow j}=\delta_{ij}A_{e,i}.
\tag{24}
\]

**证明。** 常量端口场满足 $\partial_jD_{e,i}=0$，代入式 (6)、(7) 和 (13)。$\square$

“输出函数 $F$ 是线性的”本身只保证完整切面上的 $\sum_eR_e=0$，不保证每个端口的 $R_e=0$。把两个含非线性但相互抵消的并行分支实现为同一个线性 $F$，可以得到非零且互相抵消的端口 $R$。因此式 (24) 的准确条件是端口场常量或仿射图，而不是只检查输出公式。

### 命题 6：一维反对称退化

当 $d=1$ 时，每个端口都满足

\[
\Omega_{e,11}=\partial_1D_{e,1}-\partial_1D_{e,1}=0.
\tag{25}
\]

一维中 $R$ 仍可非零；消失的是反对称曲率，不是所有二阶形成项。

### 命题 7：完整模型梯度场退化

把合法切面聚合为单一总端口后，$D_{\mathrm{total}}=\nabla F$，由 Hessian 对称性有 $\Omega_{\mathrm{total}}=0$。因此 $\Omega$ 只诊断完整梯度在端口之间如何形成相反的非保守分量；聚合后必然丢失该信息。

### 反例 1：不合法切面不能恢复 Endpoint

若 residual 图只保留 branch use 而遗漏 identity use，则 branch 路径上的逐端口式 (11)仍成立，但式 (1) 失败，因而式 (12)、(14) 和 (17) 均不能由本文定理推出。加入两个串联切面则会重复计算同一微分路径，结论同样失败。

### 反例 2：非光滑路径不能套用经典二阶链

跨越 ReLU kink 时 $D$ 可跳变，$Q=(u-b)D$ 可能不绝对连续。仅在几乎处处使用普通 Hessian 会漏掉跳跃的分布项；求积节点没有命中 kink 也不能恢复定理 1。此时应签发 `FAIL_NONSMOOTH_NODE_HIT` 或 `FAIL_QUADRATURE_CLOSURE`，而不是把结果视为本文 $C^2$ 定理的近似实例。

### 反例 3：$R$ 和 $\Omega$ 不是天然因果效应

对一维线性模型 $F(u)=u$，仿射图上 $R=0$ 且 $\Omega=0$，但把唯一内部 use 替换为零会改变输出。反过来，乘法图的单端口 $\Omega$ 可非零，但它只描述该端口场的反对称导数。梯度诊断值既不是 activation replacement 的有限差，也没有指定干预分布、替换值或结构方程。因果主张必须由另行冻结的 patch、ablation 或输入干预直接检验。

## 7. 适用边界、范围判定与可证伪性

### 7.1 已证明范围

核心式 (11)–(17) 在第 1.1 节的固定 $C^2$ 图范围内完整证明，不依赖直线路径；式 (16) 才专属于直线路径。该限制与冻结的 smooth CPU fixtures 及计划中的 `eval()`、固定 ViT 图相容，因此本步骤不触发 `NO_GO_GENERALITY`。

### 7.2 范围收缩

- 若路径完整落在一个固定激活区域且端口场在其邻域为 $C^1$，可在该区域应用定理。
- 若路径跨越 kink、图分支边界、离散路由或 shape 变化点，本文的经典二阶结论不覆盖该样本；需要显式的广义导数/分布项新合同，不能静默沿用。
- 若只能证明某个 branch cut 而非全模型合法切面，则只保留逐端口式 (11) 和分解式 (13)，不声明完整 Endpoint、差分或曲率零和。
- 若端口无法按 consumer、slot 和 invocation 唯一分离，则不定义端口级对象。

### 7.3 命题状态

| 对象 | 状态 | 直接否证条件 |
|---|---|---|
| 逐端口 PEFA 闭合，式 (11) | `PROVED` | 在满足第 1.1 节条件时出现解析反例 |
| 完整 Endpoint 闭合，式 (12) | `PROVED` | 合法切面和光滑条件均成立时不闭合 |
| $T=\delta A+R$，式 (13) | `PROVED` | 乘积导数在定义域内不成立 |
| Endpoint–IG 差，式 (14) | `PROVED` | 合法切面上出现精确反例 |
| 完整切面 $\Omega$ 零和，式 (17) | `PROVED` | 邻域切面合法且 $F\in C^2$ 时出现非零和 |
| $R/\Omega$ 对干预的预测力 | `UNTESTED_EMPIRICAL_CLAIM` | 留待步骤 5 的预注册干预评价 |
| $R/\Omega$ 的端口联合合同有增量效用 | `UNTESTED_EMPIRICAL_CLAIM` | 留待步骤 4–5 的端口与基线比较 |

本步骤没有未证明却被当作结论使用的 conjecture。经验主张保持 `UNTESTED`，不由数学闭合代替。

## 8. 与步骤 2 解析 fixture 的数值观察

本节是实现观察，不参与前述证明。独立符号程序从 `fixture_registry.json` 重建标量图和逐端口 $D$，用 SymPy 求导并用高精度一维积分计算路径量，再与步骤 2 的独立闭式 fixture 比较。

| 观察 | 步骤 2 / 步骤 3 证据 | 理论对应 |
|---|---|---|
| `LINEAR3` 的 $R$ 与 $\Omega$ 为零 | 仿射端口场逐元素为零 | 命题 5 |
| `BILINEAR2` 有非零 off-diagonal $R$，且 Endpoint–IG 差为 $(-3,-3)$ | 解析闭式、自动微分和独立符号积分一致 | 定理 2–3 |
| `FANOUT2` 必须分离两个 consumer use | 合并余切触发 `FAIL_PORT_ID_AMBIGUOUS` | 第 1.1 节端口分离 |
| `RESIDUAL2` 漏/重 identity 均失败 | 路径交点计数分别出现 0/2 | 引理 1、反例 1 |
| `MULTIINPUT3` 交换 slot 失败 | 端口身份验证失败 | 第 1.1 节端口分离 |
| `RELU_SAFE/RELU_HIT` 不进入主 PASS | 分别触发闭合失败/节点命中失败 | 反例 2 |

机器可读结果写入 `step3_symbolic_validation.json`；`validate_step3.py` 逐 fixture 检查符号恒等式、逐点导数、路径张量、变换反例、前置步骤绑定和本文哈希。

## 9. 步骤 3 出口

```text
STEP3_THEORY_STATUS=PROVED_UNDER_FROZEN_C2_GRAPH_SCOPE
CORE_IDENTITIES_PROVED=4/4
REPRESENTATION_CASES_RESOLVED=7/7
CONJECTURES_USED_AS_CONCLUSIONS=0
NO_GO_GENERALITY=false
SCIENTIFIC_EFFECT_CLAIMS=UNTESTED
STEP4_REQUIRES_EXPLICIT_USER_REQUEST
```

步骤 3 只建立理论链与适用范围。它不授权加载 ViT、读取 JPEG、使用 GPU 或启动步骤 4。
