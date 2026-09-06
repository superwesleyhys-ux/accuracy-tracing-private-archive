# 原版与 Accuracy 重测：被 API 额度阻断

日期：2026-09-05。**本次没有产生新的模型答案、准确率或稳定性结论。**

已按上一轮完全相同的源码、适配器、五条题目、gold、模型和预算重新启动测试。两次 `config.json` 完全相同，推理前已核对所有源码及数据 SHA-256；没有复用旧响应。原计划为每个系统每题重新执行一次，共十个任务。

| 系统 | 计划任务 | 成功得到答案 | 首次请求失败 | 本次准确率 |
|---|---:|---:|---:|---|
| 原版 | 5 | 0 | 5 | 无法计算 |
| Accuracy v0.3 | 5 | 0 | 5 | 无法计算 |

十个任务的第一次模型请求均返回 `RateLimitError`，没有取得模型 response ID、答案或服务端 token usage。随后仅执行了一次独立的小请求，专门区分短期限流与不可恢复的本次额度问题，接口返回：

```json
{
  "http_status": 429,
  "api_code": "credit_balance_exhausted",
  "api_type": "insufficient_quota",
  "retry_after": null
}
```

这确认当前调用因 API 额度耗尽而被拒绝。已停止重试。十次基准请求和一次诊断请求均保留记录；诊断请求不属于评分样本。

失败发生在获得任何语义答案之前，不能记作答错，也不能把失败映射成 `unverifiable` 后与 gold 比较。新分数和跨次标签稳定率均为 `null`。日志中的本地 token 计数为零仅表示没有取得 usage，不是对账单的测量。

上一次成功对比的原版 5/5、Accuracy 5/5 仍只是上次五条已知合成样例的结果；本轮没有重现或推翻它。恢复当前 API 项目的可用额度后，才可重新执行真实模型比较。

记录：[repeat-summary.json](direct-v03-fresh-02/repeat-summary.json)、[接口诊断](direct-v03-fresh-02/infrastructure-diagnosis.json)、[所有任务](direct-v03-fresh-02/results.json)、[配置一致性记录](direct-v03-fresh-02/repeat-contract.json)、[执行团队复核](direct-v03-fresh-02/AUDIT.md)。原始 `*-calls.json` 和失败报告保存在同一目录；原版及 Accuracy 的既有实现均未改动。

已有执行源码归档及完整协议见 [上次对比记录](direct-v03-fresh-01/PROTOCOL.md)。后续执行应使用新目录，保留本次失败，不覆盖为成功结果。
