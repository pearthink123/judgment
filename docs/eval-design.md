# Evaluation Framework — 评估设计

## 核心问题

judgment engine 声称能检测 Agent 故障、减少无效执行。但**没有任何数字证明**。

## 实验设计

### 对比组

| 组 | 配置 | 说明 |
|---|---|---|
| **Baseline** | Pure agent loop, no judgment | ReAct 循环照常跑，不管观测到多少次失败 |
| **Judgment** | Agent loop + DecisionEngine | 每一 step 后 engine.step(obs)，引擎说 escalate 就停 |

### 评估指标

| 指标 | 含义 | 怎么算 |
|---|---|---|
| **Task success rate** | 完成任务的比例 | completed / total |
| **Mean steps to complete** | 成功任务的平均步数 | Σ steps_of_successful / n_successful |
| **Wasted steps ratio** | 失败任务中浪费的步数占比 | steps_taken / max_steps (仅失败任务) |
| **Detection precision** | 报警后确实有故障的比例 | true_alarms / total_alarms |
| **Detection recall** | 故障被检测到的比例 | detected_faults / total_faults |
| **Mean detection delay** | 从故障注入到首次报警的步数 | Σ (alarm_step - fault_step) |
| **False escalation rate** | 健康轨迹被错误中断的比例 | n_healthy_escalated / n_healthy |

### 故障模型（4种 realistic pattern）

不是简单的"随机失败"——模拟真实 Agent 的退化模式：

1. **Context Drift** — LLM 输出逐渐跑偏，工具调用还是"成功"的但语义偏离
   - 模拟方式：progress_delta 逐渐递减（0.15 → 0.08 → 0.02 → 0），但 tool_ok 始终=true
   - **最难检测**——因为工具调用一直"成功"

2. **Tool Degradation** — 工具调用从正常衰落到频繁失败
   - 模拟方式：前 5 步正常，之后 tool_ok 概率从 0.95 线性降到 0.20
   - **中等难度**——信号明确但渐进

3. **Loop Trap** — Agent 陷入重复循环（同一工具反复调用）
   - 模拟方式：连续 5+ 步调用同一工具，progress=0，错误计数不变
   - **中等难度**——需要 cumulative 检测

4. **Catastrophic Cascade** — 一步失败→上下文污染→后续全部失败
   - 模拟方式：step 5 注入一次"灾难性"失败，之后 80% 步都失败
   - **最容易检测**——信号最强

### 场景分配

每种故障模式 25 条轨迹 + 25 条健康轨迹 = 125 条 × 2 组（baseline + judgment）= 250 条轨迹

## 实现计划

```
scripts/
  eval_runner.py        # 对比实验主入口
  fault_models.py        # 4 种 realistic 故障生成器
```

输出：
- `scripts/eval_results.json` — 原始数据
- `scripts/eval_report.md` — 人类可读报告
