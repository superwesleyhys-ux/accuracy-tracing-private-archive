# Accuracy Tracing v0.3：离线验收记录

生成时间：2026-09-06T06:07:20.228441+00:00

本报告描述当前源码的离线测试、安装烟测，以及 historical-2023 真实命题语料的合同审计。没有调用新闻检索服务或模型 API，没有执行 live staged/monolithic A/B，也没有测得真实准确率。

## 运行结果

- 测试：198 项；失败 0；错误 0；跳过 0。
- 核心 wheel 安装检查：passed。
- staged 分阶段运行时：源码 checkout 导入/CLI 帮助烟测；本次没有执行模型推理。

| 测试组 | 数量 |
|---|---:|
| test_comparison | 6 |
| test_evaluation | 19 |
| test_harness | 25 |
| test_historical_compare | 22 |
| test_provenance | 26 |
| test_release | 4 |
| test_repair | 48 |
| test_semantic_adapter | 16 |
| test_staged_semantic | 32 |

## Historical-2023 合同审计

- 状态：`passed`；语料：8 条 2023 年真实世界命题。
- 标签：true 4 条 / false 4 条。
- 推理证据截止：`2024-12-31T23:59:59Z`。
- 审计产物：`reports/historical-2023-audit-v0.3.json`。

这 8 条语料使用声明为官方来源的命题与定论材料；本次检查覆盖时间窗、角色隔离、内部内容哈希、gold 隔离及冻结校验和等合同。它不会重新抓取并独立认证远端原件，也没有运行任何模型，因此不能产生 staged/monolithic live A/B 或准确率结论。

## 双回环演示

- 检索轮次：3。
- 不同材料版本：4。
- 分解调用：6。
- 验证调用：3。
- 演示状态：`original_material_located` / `contradicted` / `complete`。

演示使用程序化 fixture 标注，只验证路由、顺序、预算、状态与审计合同。它不是通用语义模型实测，也不证明拆分提示词提升了准确率。

## 指标算术样例

以下分数属于四条故意含错的手写预测，只检查评分器，不代表项目或任何模型的性能。

| 指标 | 手写样例计算值 |
|---|---:|
| VP | 0.500000 |
| FR | 0.500000 |
| TR | 0.500000 |
| SR | 0.666667 |
| EN | 0.666667 |
| CA | 0.656250 |
| HFAR | 0.500000 |

## 仍需外部验证

- 真实新闻检索适配器，以及对 historical-2023 远端来源原件的独立重抓取/认证。
- 独立人工 gold、隐藏且按事件/时间隔离的真实新闻测试集。
- 等总预算 staged/monolithic live A/B；当前离线合同审计不能推导真实准确率增益。
- 生产并发、网络故障、成本和长期稳定性测试。
