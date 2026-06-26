# Agent Decision Engine — 架构重构设计

## 0. 问题定义

Agent harness 的核心决策问题：

> 给定截至步骤 t 的全部观测历史 ℋ_t = {o_1, o_2, ..., o_t}，Agent 应该继续自主执行、采取纠正措施、还是请求用户介入？

这本质上是**部分可观马尔可夫决策过程** (POMDP)，但我们不直接求解完整 POMDP（维度灾难），而是用一个**三层堆叠架构**逐层逼近最优策略。

---

## 1. 三层架构总览

```
观测流 ℋ_t
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Layer 1: 异常检测 (CUSUM + Hawkes 似然)          │
│   - 实时监控观测序列的 "意外程度"                  │
│   - Hawkes 提供基线似然结构 (正常的事件聚集不应报警) │
│   - 输出: 异常信号 s₁ ∈ ℝ (累积漂移量)            │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Layer 2: 隐状态估计 (3 状态 HMM)                  │
│   - 状态空间: {Healthy, Degraded, Broken}        │
│   - 观测: 工具成败, 进度增量, 用户交互, 误差累积    │
│   - 输出: 信念向量 b = [P(H), P(D), P(B)]        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Layer 3: 阈值门控决策                             │
│   - P(B) > θ_B      → Escalate                  │
│   - P(D) > θ_D      → Correct                   │
│   - P(H) > θ_H      → Continue                   │
│   - Ambiguous       → Gather Info (最小成本行动)   │
└─────────────────────────────────────────────────┘
```

**为什么不直接做完整 POMDP 求解？**
完整 POMDP 的状态空间 = 隐状态 × 信念单纯形，行动空间 ~10，观测空间 ~∞。精确求解不可行。三层堆叠是**结构化近似**：Layer 1 做快速统计检测，Layer 2 做概率推断，Layer 3 做确定性门控——每层都有封闭的数学基础，复合后仍然可解释。

---

## 2. Layer 1: CUSUM 异常检测 + Hawkes 似然

### 2.1 CUSUM 基础

CUSUM (Cumulative Sum) 检测一个受监控的统计量何时从"受控"值 μ₀ 漂移到"失控"值 μ₁。两个关键递归：

#### 标准形式 (Page, 1954)

给定观测序列 X_t ~ f₀ (受控) 或 f₁ (失控)：

```
S_t = max(0, S_{t-1} + L_t)      S_0 = 0
L_t = log [ f₁(X_t) / f₀(X_t) ]   ← 对数似然比
```

当 `S_t > h` (阈值) 时报警。报警后 S_t 重置为 0。

#### 对 Agent 的适配

我们不检测"观测分布本身变了"，而检测"观测序列与健康模型的偏离程度"：

```
L_t = -log P_healthy(o_t | ℋ_{t-1}) + log P_reference(o_t)
     = surprisal_healthy(o_t) - surprisal_reference(o_t)
```

- `P_healthy` = 假定系统健康时，观测 `o_t` 的似然（由 Hawkes 提供结构 + 观测模型提供分布）
- `P_reference` = 一个"无信息"基准分布（均匀分布或经验分布）

直觉：**在健康状态下，每个观测都有一定程度的预期；连续出现"意外"观测 → 累积漂移上升 → 报警。**

### 2.2 Hawkes 的角色：提供基线似然结构

纯 CUSUM 的问题：如果健康状态下工具调用天然呈聚集模式（连续 3 次 read_file 很正常），纯独立假设会误报。

Hawkes 修正：**观测似然不是独立同分布的**——过去事件通过 Hawkes 自激影响当前观测的期望。所以：

```
P_healthy(o_t | ℋ_{t-1}) = P_obs_type(o_t | λ_t^{(H)})
```

其中 `λ_t^{(H)}` 是假定系统在 Healthy 状态时所有维度的事件强度。我们不需要完整的 4×4 互激发矩阵——只需要 **Healthy 基线 Hawkes 的似然评估**：

- 给定历史，预期的工具调用频率是多少？
- 在当前历史下，一个新的 error 观测有多"意外"？

**降级的 Hawkes**：不再输出"该不该行动"的决策，而是输出**"这个观测在当前历史下有多意外"的标量分数**。

### 2.3 完整 Layer 1 算法

```
Input: 观测序列 o_t = (tool_ok, progress_delta, has_user_msg, error_count_delta)
Output: 漂移累积量 s_t

1. 从 Hawkes 计算当前条件下各观测的期望似然: L_hawkes = log λ_d(o_t | ℋ_{t-1})
2. 计算观测意外度: surprise = -log P_obs(o_t | Healthy 模型) + log P_obs(o_t | Uniform)
3. 加入 Hawkes 修正: L_t = surprise - γ · L_hawkes  
   (γ 控制 Hawkes 修正的强度；如果 Hawkes 预期到这个事件，surprise 被削弱)
4. S_t = max(0, S_{t-1} + L_t)
5. 如果 S_t > h: emit 异常信号, S_t = 0
```

参数：
- `μ₀` = 0 (受控均值，L_t 的期望在 H₀ 下为负)
- `h` = 报警阈值 (典型值 3.0–5.0，控制误报/漏报权衡)
- `γ` = Hawkes 修正强度 (0.2–0.5)

---

## 3. Layer 2: 三状态 HMM

### 3.1 状态定义

```
Healthy  (H): Agent 在正常轨道上
  - 工具成功率 ~75-85%
  - 每次步骤有正向进度
  - 偶尔的失败是正常的
  - 用户交互稀疏且通常为确认而非纠正

Degraded (D): 系统出现局部问题但可能自愈
  - 工具成功率 ~40-60%
  - 进度缓慢或停滞
  - 错误有聚集趋势
  - 可能自行恢复 (HDD 循环)

Broken   (B): Agent 已经陷入无效循环
  - 工具成功率 <30%
  - 进度为零或为负
  - 连续相似错误
  - 需要外部干预
```

### 3.2 状态转移矩阵 (先验)

```
        H       D       B
H   [ 0.80   0.17   0.03 ]   ← Healthy 大概率保持
D   [ 0.15   0.65   0.20 ]   ← Degraded 可能自愈或恶化
B   [ 0.02   0.10   0.88 ]   ← Broken 很难自愈
```

设计依据：
- 对角占优（Markov 惯性）
- H→D 概率 (~0.17) 高于 D→H (~0.15)，系统中存在熵增趋势
- B 的吸收性 (~0.88 对角)：一旦进入 Broken，自愈概率很低
- 这些先验会在在线推理中被观测证据修正

### 3.3 观测模型

我们将连续观测离散化为类别，用于 HMM 发射概率。维度：

#### 维度 1: 工具成败 (2 类)
```
              H      D      B
tool_ok    0.80   0.50   0.25
tool_fail  0.20   0.50   0.75
```

#### 维度 2: 进度增量 (3 类)
```
                H      D      B
progress_pos  0.75   0.35   0.10
progress_zero 0.20   0.55   0.60
progress_neg  0.05   0.10   0.30
```

#### 维度 3: 用户交互 (2 类)
```
              H      D      B
user_msg     0.05   0.15   0.25
user_silent  0.95   0.85   0.75
```

#### 维度 4: 误差累积趋势 (2 类)
```
              H      D      B
err_stable   0.90   0.50   0.20
err_rising   0.10   0.50   0.80
```

乘积似然（假设维度条件独立给定隐状态）：
```
P(o_t | state) = ∏_{dim} P(o_t^dim | state)
```

这是 HMM 的标准做法；维度独立性假设在实践中效果良好且使推理 tractable。

### 3.4 前向算法 (Forward Algorithm)

定义 `α_t(s) = P(S_t = s, o_{1:t})`：

```
初始化: α₁(s) = π_s · P(o₁ | S₁ = s)

递推:   α_t(s) = P(o_t | S_t = s) · Σ_{s'} α_{t-1}(s') · T_{s'→s}

后验:   P(S_t = s | o_{1:t}) = α_t(s) / Σ_{s'} α_t(s')
```

数字稳定化：使用对数域计算，每步做 log-sum-exp 归一化。

### 3.5 参数可学习性

如果有标注数据（Agent 运行日志 + 人工标注的状态标签），用 Baum-Welch (EM) 学习 T 和发射矩阵。
没有标注数据时，使用上述领域知识先验 + 在线推理即可——先验足够编码正确的相对顺序。

---

## 4. Layer 3: 阈值门控决策

### 4.1 决策规则

给定信念向量 `b = [P(H), P(D), P(B)]`：

```
IF P(B) >= θ_B  (建议 0.45):
    → ESCALATE: 请求用户介入
    → 行动: ask_user / rollback / restart_task

ELSE IF P(D) >= θ_D  (建议 0.35):
    → CORRECT: 采取纠正措施
    → 行动: verify / run_tests / rethink_plan

ELSE IF P(H) >= θ_H  (建议 0.60):
    → CONTINUE: 继续当前计划
    → 行动: 下一个计划中的工具调用

ELSE (ambiguous):
    → GATHER: 收集更多信息
    → 行动: 成本最低的信息获取行动 (read_file, check_status)
```

### 4.2 阈值校准

阈值之间的间隔 (θ_D < θ_B, θ_H 独立于前两者) 形成了**滞回区**，防止状态在边界附近振荡。

| 阈值 | 推荐值 | 含义 |
|---|---|---|
| θ_B | 0.45 | P(B) ≥ 45% → 必须干预 |
| θ_D | 0.35 | P(D) ≥ 35% → 谨慎纠正 |
| θ_H | 0.60 | P(H) ≥ 60% → 安全自主 |

### 4.3 滞回保护

为防止阈值附近反复跳变 (边界震荡)，引入滞回：
```
如果上一步状态 != 当前候选状态:
    当前候选需要超出阈值 Δ 额外 margin
    Δ = 0.08 (建议值)
```

### 4.4 行动选择细节

当 Layer 3 输出 `CORRECT` 时，具体采取什么纠正行动取决于观测上下文：

- 如果最近 3 步有 2 次工具失败 → `verify`（检查当前状态）
- 如果连续出现同类错误 → `rethink`（重新规划方法）
- 如果进度为零超过 5 步 → `read_file`（重新理解上下文）

这是**简单启发式分支**——远少于原版的 160 行假数学，明确标注为启发式。

---

## 5. 完整决策循环

```
每个步骤 t:
    观测 o_t = 来自 harness 的原始信号

    # ---- Layer 1: 异常检测 ----
    L_t = compute_surprisal(o_t, hawkes_baseline)
    S_t = max(0, S_{t-1} + L_t)
    anomaly = (S_t > h)
    if anomaly: S_t = 0  (重置)

    # ---- Layer 2: 信念更新 ----
    α_t = forward_step(α_{t-1}, o_t)   (online HMM filtering)
    b_t = α_t / Σ α_t

    # ---- Layer 3: 决策 ----
    action = threshold_gate(b_t, prev_action)
    
    # 如果需要采取行动 (非 CONTINUE):
    #   action 被注入 agent 的主循环
    
    return action, b_t, S_t
```

---

## 6. 与旧架构的对比

| 维度 | 旧架构 (假数学) | 新架构 (三层堆叠) |
|---|---|---|
| 状态估计 | 四个独立概率的启发式累加 | 3 状态 HMM Forward 滤波，真正的贝叶斯推断 |
| 异常检测 | 无 | CUSUM + Hawkes 似然，统计检测理论保证 |
| 决策 | 手工 if-else × 假 EVOI | 阈值门控 + 滞回，可解释边界 |
| 数学严谨性 | 命名与实际不符 | 每层有封闭数学 + 可引用文献 |
| 可调试性 | 魔法数字遍布 | 每层输出可独立可视化 |
| 可扩展性 | 加规则 = 加 if | 加状态/加观测维度 = 参数扩展 |

---

## 7. 文件结构 (目标)

```
core/
  hmm.py           # HMM: 状态定义, 转移矩阵, 发射矩阵, Forward 算法
  cusum.py         # CUSUM: 对数似然比计算, 累积漂移, 阈值报警
  hawkes.py         # 降级 Hawkes: 仅提供基线似然 (不再做决策)
  engine.py         # DecisionEngine: 三层堆叠集成, 统一 API
  diagnostics.py    # 诊断输出, 可视化数据生成
```

旧的 `bayesian.py`, `info_gain.py`, `control.py`, `mdp_pomdp.py` → 删除或归档。

---

## 8. 验证计划

### 8.1 单元测试

| 测试 | 验证内容 |
|---|---|
| `test_hmm_forward` | 连续 error → P(B) 单调递增; 连续 success → P(H) 恢复 |
| `test_cusum_no_false_alarm` | 正常观测序列 50 步不触发 S_t > h |
| `test_cusum_detection` | 注入 5 个连续异常观测，S_t 在 3-4 步内突破 h |
| `test_threshold_hysteresis` | 信念在边界振荡时不产生行动翻转 |
| `test_hawkes_likelihood_falloff` | e^{-β·Δt} 正确衰减 |

### 8.2 集成测试

| 测试 | 验证内容 |
|---|---|
| 正常执行轨迹 | 20 步顺利执行，始终 CONTINUE |
| 错误注入 | 连续 3 次 tool fail → 触发 CORRECT |
| 错误级联 | 持续错误 → Broken → ESCALATE |
| 自恢复 | Degraded + 后续 success → 恢复 Healthy |
| 用户干预 | 用户消息 → 信念调整 → 恢复或升级 |

---

## 9. 参考文献

- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2), 100–115.
- Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models." *Proceedings of the IEEE*, 77(2), 257–286.
- Hawkes, A. G. (1971). "Spectra of Some Self-Exciting and Mutually Exciting Point Processes." *Biometrika*, 58(1), 83–90.
- Tartakovsky, A., Nikiforov, I., & Basseville, M. (2014). *Sequential Analysis: Hypothesis Testing and Changepoint Detection*. CRC Press.
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). "Planning and Acting in Partially Observable Stochastic Domains." *AIJ*, 101(1-2), 99–134.
