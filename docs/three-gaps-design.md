# 三个缺口 — 设计文档

## 缺口 1: 纠正行动选择器

### 问题

Layer 3 输出 CORRECT 时，只说了"该纠正了"，没说怎么纠正。需要第二级决策。

### 方案

纠正行动分类：

| 行动 | 含义 | 触发条件 |
|---|---|---|
| `verify` | 检查当前状态（读文件、跑诊断） | 最近 3 步有 ≥2 次工具失败 |
| `rethink` | 重新规划方法（换策略、退一步） | 同类错误重复出现；或进度=0 持续 ≥5 步 |
| `retry` | 重试上次失败的操作 | 单次孤立失败，非重复 |
| `rollback` | 撤销上一步变更 | 连续失败且观测到负进度 |

设计原则：
- **明确标注为启发式** — 不包装成数学推导
- 每个规则有清晰的触发条件和操作性含义
- 如果 POMDP Layer 3 直接输出最优行动（缺口 2），则本模块退化为 fallback

---

## 缺口 2: POMDP 价值迭代 → 替代阈值门控

### 问题

当前 Layer 3 的阈值门控是近似，不是理论最优。给定明确的代价函数，POMDP 能从第一性原理给出最优策略。

### 为什么现在能做

3 状态 × 4 行动 = 极小的状态空间。信念单纯形在分辨率 0.05 下只有 231 个网格点，**精确价值迭代完全可行**（毫秒级）。

### 数学模型

**信念 MDP**：把 POMDP 转化为定义在连续信念空间上的 MDP。

贝尔曼最优方程：

```
V*(b) = max_a [ Σ_s b(s)·R(s,a) + γ · Σ_o P(o|b,a) · V*(b') ]

其中 b'(s') = P(s'|o,a,b) ∝ P(o|s',a) · Σ_s T(s'|s,a) · b(s)
```

**值迭代算法**（离线，在信念单纯形网格上）：

```
1. 初始化 V₀(b) = 0 对所有信念点 b ∈ Δ
2. 重复直到收敛：
   V_{k+1}(b) = max_a [ Σ_s b(s)·R(s,a) + γ · Σ_o P(o|b,a) · V_k(SE(b,a,o)) ]
   其中 SE(b,a,o) 是状态估计器（贝叶斯更新）
3. 输出策略：π*(b) = argmax_a Q(b,a)
```

**运行时**：给定当前 HMM 信念 b，找最近的网格点，返回 π*(b)。

### 奖励函数（需要你确认）

```
R(s, a):
            Healthy   Degraded   Broken
continue    +1        −1         −8       ← 坏了还继续 = 灾难
correct     −0.5      +3         +0.5     ← 纠正帮降级，浪费在健康上
escalate    −4        −0.5       +6       ← 求救很贵，但坏了必须
gather      +0.2      +1.5       −2       ← 信息收集帮降级，坏了时耽误
```

设计直觉：
- **continue × Broken = −8**（最大惩罚）：Agent 已经坏了还在瞎跑，每步都在浪费上下文和积累错误
- **escalate × Healthy = −4**（次级惩罚）：不必要的打断用户体验很差
- **correct × Degraded = +3**（最大奖励）：在问题恶化前纠正
- γ = 0.90

### 与阈值门控的关系

POMDP 价值迭代 **替代** Layer 3 的阈值门控。阈值门控保留为 fallback（当 POMDP 求解器不可用时）。

---

## 缺口 3: Baum-Welch 参数学习

### 问题

HMM 的转移矩阵和发射概率目前是手工设定的。如果有 Agent 运行日志，应该从数据中学习。

### 数学模型

**Baum-Welch = EM for HMM**（Rabiner 1989, §III-C）。

给定观测序列集合 {O^(1), O^(2), ..., O^(N)}：

**E 步**（用当前参数 λ = (π, T, B) 计算）：

Forward:  α_t(i) = P(o_1...o_t, S_t=i | λ)
Backward: β_t(i) = P(o_{t+1}...o_T | S_t=i, λ)

```
ξ_t(i,j) = P(S_t=i, S_{t+1}=j | O, λ)
         = α_t(i)·T_{ij}·B_j(o_{t+1})·β_{t+1}(j) / P(O|λ)

γ_t(i)   = P(S_t=i | O, λ) = Σ_j ξ_t(i,j)
```

**M 步**（更新参数）：

```
π_i'   = γ_1(i)
T_ij'  = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
B_j(k)' = Σ_t γ_t(j)·1[o_t=k] / Σ_t γ_t(j)
```

### 实现要点

- **全 log-space** — 防止下溢
- **多序列支持** — 聚合所有序列的期望计数
- **收敛检测** — log-likelihood 变化 < 1e-4 时停止
- **半监督模式** — 如果某些时刻有人工标注的状态标签，固定 γ_t = 1 对该状态

### 输入格式

```python
# 每条轨迹是观测类别序列
sequences = [
    # 轨迹 1: 顺利执行 10 步
    [obs1, obs2, ..., obs10],
    # 轨迹 2: 出错后恢复
    [obs1, obs2, ..., obs15],
    ...
]

# 可选：部分状态标签
labels = [
    {5: STATE_DEGRADED, 6: STATE_BROKEN},  # 轨迹 1 的标签
    {},                                      # 轨迹 2 无标签
]

T_learned, B_learned, pi_learned, ll_history = baum_welch(sequences, labels)
```

---

## 实现计划

| 序号 | 文件 | 内容 |
|---|---|---|
| 1 | `core/corrective.py` | CorrectiveSelector：4 条启发式规则 |
| 2 | `core/pomdp.py` | belief-MDP 值迭代 + 运行时策略查找 |
| 3 | `core/training.py` | Baum-Welch + Forward-Backward |

| 序号 | 文件（修改） | 变更 |
|---|---|---|
| 4 | `core/engine.py` | Layer 3 用 POMDP 策略替代阈值门控；集成 CorrectiveSelector |
| 5 | `tests/test_corrective.py` | 每条规则的触发/不触发测试 |
| 6 | `tests/test_pomdp.py` | 值迭代收敛、策略合理性、Belief backup |
| 7 | `tests/test_training.py` | 合成数据恢复真参数、多序列、log-likelihood 单调 |

---

## 需要你确认

### Q1: 奖励函数的数值

上面的 R(s,a) 表你觉得合理吗？continue×Broken=−8 够重吗？escalate×Healthy=−4 是否太贵或太便宜？

### Q2: POMDP 的观测模型

当前 HMM 的观测不依赖于行动（P(o|s) 不涉及 a）。要不要让 POMDP 的观测也依赖于行动？还是保持独立（P(o|s,a) = P(o|s)）？

保持独立的好处：当前 HMM 发射概率直接复用。
依赖行动的好处：更精确（例如 escalate 之后用户回复的概率不应该等于 continue 时用户回复的概率）。

我建议**先用独立**，跑通了再扩展。

### Q3: Baum-Welch 要不要做半监督？

纯无监督学习的 HMM 参数可能学到"没用"的结构（比如把 Healthy 和 Degraded 的语义互换）。半监督（少数标签锚定语义）更安全。

我建议做半监督——即使只有 2-3 个标签点也能锚定状态语义。

---

你审一下，确认后我开写。
