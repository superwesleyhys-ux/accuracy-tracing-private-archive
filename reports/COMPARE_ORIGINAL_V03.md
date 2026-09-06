# 原版与 Accuracy v0.3：重新执行的直接对比

日期：2026-09-05。结果：**同一组 5 条合成样例，原版 5/5，Accuracy 5/5，标签正确数打平。**
这是两套流水线的全新执行，不是上次修复版内部的首轮检查点对比，也不是将 Accuracy 接在原版输出之后的增补实验。

## 逐题结果

| 固定问题 | 标准标签 | 原版 | Accuracy |
|---|---|---|---|
| 公告是否支持“桥梁已经重开” | false | false | false |
| 公告是否支持“图书馆改为晚上九点闭馆” | true | true | true |
| 三跳引用链终点是否支持“实验室八点开放” | true | true | true |
| 两份渡轮报道说法相反 | disputed | disputed | disputed |
| 合成公告能否证实现实中的图书馆营业时间 | unverifiable | unverifiable | unverifiable |

没有出现只由 Accuracy 答对的题，也没有出现原版正确、Accuracy 改错的题。双方均完成全部 5 题，接口或有效性错误均为 0。

## 实际消耗

| 指标（五题合计） | 原版 | Accuracy | Accuracy 相对减少 |
|---|---:|---:|---:|
| 模型调用数 | 81 | 26 | 67.9% |
| 输入 tokens | 72,841 | 38,502 | 47.1% |
| 输出 tokens | 42,152 | 18,368 | 56.4% |
| 输入加输出 tokens | 114,993 | 56,870 | 50.5% |
| 各任务耗时之和 | 796.00 秒 | 326.60 秒 | 59.0% |

共取得 **107 个不同的真实 API response ID**，返回模型均为 `gpt-6-astra`，没有响应缓存复用或自动重试。耗时是两个任务并发运行时的观测值，各任务耗时之和不是整个批次的墙钟时间。这里报告 tokens，不把它们当成相同价格的计费单位。

原版还生成因果树、时间线和综合报告；Accuracy 侧重材料拆解、引用关系与分层判断。因此，较低消耗只适用于本轮窄任务，不能解释成同等全部产品功能下的性能领先。

## 公平性与版本

- 原版源码：`news-tracking-perdiction@789506115e7dfef1f2359c2e43bcae4234b8d3f2`，`NewsTracingAgent.run()` 实际执行，最大因果深度 1。五个源码文件与之前固定的源码指纹相同。
- Accuracy 引擎及既有适配器：`accuracy-tracing@e36292957ce6a6cb5e5c187151681a4503302999`。15 个既有 Python 文件逐一匹配该提交的 Git blob；本轮未修改这些文件。
- 双方都使用同一份任务契约、同一套完整可用材料，从一开始即可访问全部快照。这排除了某一组单纯因为多拿了材料而获胜的因素；它不是互联网搜索质量评测。
- 每题每组上限相同：24 次调用、36,000 输出 tokens、每次 2,500 输出 tokens、600 秒。输入 tokens 记录但不设上限。相同上限不等于实际消耗相同。
- 原版搜索接口替换为只读取固定材料的模型客户端，两边都收到明确的 evidence/world 问题定义。原版算法未改，但这不是原版默认联网配置；也不能将旧测试与本次分数变化只归因于引擎代码。
- 两边使用同一套“只提取报告既有结论”的提示和输出格式。提取步骤看不到独立源材料或标准答案，不允许重新判题；提取的依据必须是报告中的原文。Accuracy 还检查提取结果是否保持其明确的原生结论。
- 逐项复核了五份原版直答与提取引用，标签与其明确结论一致；五份 Accuracy 提取标签均与原生结构化标签一致。这是执行方复核，不是独立盲评。

## 回环到底做了什么

本轮 Accuracy 五题都只进行了 **一次完成的验证判断**。桥梁、图书馆和三跳链以 `complete` 停止；渡轮冲突和现实未知以 `provider_exhausted` 停止。材料接收和引用上游到达仍会触发拆解、重新拆解；最大五轮是上限，没有为了凑轮数重复作答。

因此，这一轮没有证明多轮验证优于单轮。它说明：材料给齐、问题定义一致时，两套系统在这五个简单例子上都能正确判断；Accuracy 用更少调用完成了这个较窄的任务。

## 限制

这仍是之前公开在项目中的五条作者编写合成样例，每题每组只新采样一次。不是未见题、独立标注集或真实新闻测试，不能声称真实新闻正确率 100%、统计上的优越性或完整溯源指标全部通过。本轮量化的是四分类结论与消耗，没有伪造来源链正确率。文字报告和结构化报告经模型提取统一标签，也仍是测量限制。

## 复现与原始记录

协议：[PROTOCOL.md](direct-v03-fresh-01/PROTOCOL.md)。评分：[scores.json](direct-v03-fresh-01/scores.json)。全部结果：[results.json](direct-v03-fresh-01/results.json)。配置与指纹：[config.json](direct-v03-fresh-01/config.json)。

每题每组的 `*-calls.json` 保存全部请求、原始返回、API ID、tokens 和耗时；`*-report.json` 保存报告；`*-result.json` 保存提取原文、标签与消耗。`executed-code/` 保存实际执行源码。输入与 gold 的原始 SHA-256 未改变，推理进程不读取 gold，评分命令在十个任务全部结束后单独执行。

在安装原版及模型依赖、通过环境提供 API 凭据后：

```bash
python experiments/original_compare.py run --inputs experiments/inputs-v03.json --original /path/to/pinned-original --output reports/new-direct-run
python experiments/original_compare.py score --gold experiments/gold-v03.json --run reports/new-direct-run
```

新建结果目录，不覆盖本次记录。复现将产生新的模型调用及费用，结果允许与本次不同。
