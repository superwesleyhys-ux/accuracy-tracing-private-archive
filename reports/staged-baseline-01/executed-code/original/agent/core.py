from __future__ import annotations

import asyncio

from rich.console import Console

from . import prompts
from .llm_client import LLMClient
from .models import (
    CredibilityBreakdown,
    EventNode,
    FactClaim,
    FocusSubject,
    ForecastScenario,
    RelativeStep,
    ReplayReport,
    SourceEntry,
    StepPrediction,
    TimelineEvent,
    TracingReport,
)


class NewsTracingAgent:
    """双轨并行 + 递归因果挖掘的新闻溯源 Agent（精细化版）。"""

    def __init__(
        self,
        llm: LLMClient,
        max_depth: int = 3,
        console: Console | None = None,
    ) -> None:
        self.llm = llm
        self.max_depth = max_depth
        self.console = console or Console()

    # ══ 主入口 ══════════════════════════════════════════════

    async def run(self, news_input: str) -> TracingReport:
        # Phase 1: 解构 + 搜索规划
        self.console.print("\n[bold cyan]Phase 1: 解构 + 搜索规划[/bold cyan]")
        parsed = await self._deconstruct(news_input)

        self.console.print(f"  核心事件: {parsed.get('core_event', '?')}")
        self.console.print(f"  日期: {parsed.get('date', '未知')}")
        entities = parsed.get("entities", {})
        ent_list = entities.get("people", []) + entities.get("organizations", [])
        if ent_list:
            self.console.print(f"  关键实体: {', '.join(ent_list)}")

        queries = await self._plan_searches(parsed)
        self.console.print(f"  搜索策略: {len(queries)} 个角度")
        for q in queries:
            self.console.print(f"    - [{q['angle']}] {q['query']}")

        root_event = EventNode(
            title=parsed.get("core_event", "未知事件"),
            date=parsed.get("date", "未知"),
            summary="\n".join(parsed.get("key_claims", [])),
        )

        # Phase 2: 双轨并行
        self.console.print("\n[bold cyan]Phase 2: 双轨并行分析[/bold cyan]")

        source_result, causal_tree = await asyncio.gather(
            self._trace_sources(queries),
            self._dig_causes(root_event, depth=0),
        )
        source_pool, verify_data = source_result

        # Phase 3: 时间线构建 + 锚定验证 + 综合研判
        self.console.print(
            "\n[bold cyan]Phase 3: 时间线构建 + 综合研判[/bold cyan]"
        )

        causal_tree = await self._ground_check(causal_tree, source_pool)

        event_timeline = await self._build_timeline(source_pool, causal_tree)
        self.console.print(f"  事件时间线: {len(event_timeline)} 个事件")

        event_timeline = await self._analyze_perspectives(
            event_timeline, source_pool,
        )

        cred = self._compute_credibility(
            source_pool, verify_data, causal_tree,
        )
        self.console.print(
            f"  可信度: {cred.source_count} 信源 | "
            f"{len(cred.source_types)} 类 | "
            f"锚定 {cred.grounded_count}/{cred.total_causal_nodes}"
        )

        direct_response = await self._generate_direct_response(
            news_input, source_pool, verify_data,
            event_timeline, causal_tree, cred,
        )

        report = await self._synthesize(
            source_pool, verify_data, causal_tree,
            event_timeline, cred, direct_response,
        )

        self.console.print(f"\n[dim]{self.llm.token_summary}[/dim]")
        return report

    # ══ Replay 模式：人物视角序列 + 遮蔽预测 + +1 场景 ═══════

    async def run_replay(
        self,
        news_input: str,
        focus_person: str | None = None,
        history_steps: int = 10,
        strict_replay: bool = True,
        continue_on_mismatch: bool = False,
    ) -> ReplayReport:
        """先走完整溯源骨架，再构造相对序列、逐点遮蔽验证、+1 场景预测。"""
        self.console.print("\n[bold cyan]Replay: Phase 1 解构 + 焦点 + 搜索规划[/bold cyan]")
        parsed = await self._deconstruct(news_input)
        focus_subject = await self._select_focus_subject(parsed, focus_person)
        self.console.print(
            f"  焦点: {focus_subject.name} ({focus_subject.subject_type}) "
            f"| 置信 {focus_subject.confidence:.0%}"
        )

        queries = await self._plan_searches_with_focus(parsed, focus_subject)
        self.console.print(f"  搜索策略: {len(queries)} 个角度")
        for q in queries:
            self.console.print(f"    - [{q['angle']}] {q['query']}")

        root_event = EventNode(
            title=parsed.get("core_event", "未知事件"),
            date=parsed.get("date", "未知"),
            summary="\n".join(parsed.get("key_claims", [])),
        )

        self.console.print("\n[bold cyan]Replay: Phase 2 双轨并行[/bold cyan]")
        source_result, causal_tree = await asyncio.gather(
            self._trace_sources(queries),
            self._dig_causes(root_event, depth=0),
        )
        source_pool, _verify_data = source_result

        self.console.print(
            "\n[bold cyan]Replay: Phase 3 时间线 + 相对序列 + 逐步 replay[/bold cyan]"
        )
        causal_tree = await self._ground_check(causal_tree, source_pool)
        event_timeline = await self._build_timeline(source_pool, causal_tree)
        self.console.print(f"  事件时间线: {len(event_timeline)} 个事件")
        event_timeline = await self._analyze_perspectives(
            event_timeline, source_pool,
        )

        truth_sequence, seq_gaps = await self._build_relative_sequence(
            focus_subject,
            source_pool,
            causal_tree,
            event_timeline,
            history_steps,
        )
        if seq_gaps:
            self.console.print(f"  [yellow]序列缺口提示: {len(seq_gaps)} 条[/yellow]")

        predictions, stopped_at, accuracy = await self._run_stepwise_replay(
            truth_sequence,
            focus_subject,
            history_steps,
            strict_replay,
            continue_on_mismatch,
        )
        scenarios, forecast_summary = await self._forecast_next_step(
            truth_sequence,
            predictions,
            focus_subject,
        )

        summary_parts = [forecast_summary]
        if seq_gaps:
            summary_parts.append("序列构建备注: " + "；".join(seq_gaps[:5]))
        if stopped_at is not None:
            summary_parts.append(
                f"严格模式下于 E[{stopped_at}] 中断；准确率 {accuracy:.0%}（已完成步数内）。"
            )
        else:
            summary_parts.append(f"逐步验证完成；准确率 {accuracy:.0%}。")

        self.console.print(f"\n[dim]{self.llm.token_summary}[/dim]")
        return ReplayReport(
            focus_subject=focus_subject,
            truth_sequence=truth_sequence,
            predictions=predictions,
            strict_mode=strict_replay,
            stopped_at=stopped_at,
            next_step_scenarios=scenarios,
            accuracy=accuracy,
            summary="\n".join(summary_parts),
        )

    async def _plan_searches_with_focus(
        self, parsed: dict, focus_subject: FocusSubject,
    ) -> list[dict]:
        plan_input = (
            f"核心事件: {parsed.get('core_event', '')}\n"
            f"日期: {parsed.get('date', '未知')}\n"
            f"人物: {parsed.get('entities', {}).get('people', [])}\n"
            f"组织: {parsed.get('entities', {}).get('organizations', [])}\n"
            f"地点: {parsed.get('entities', {}).get('locations', [])}\n"
            f"关键声明:\n"
            + "\n".join(f"- {c}" for c in parsed.get("key_claims", []))
            + "\n\n=== Replay 视角锚点 ===\n"
            f"焦点名称: {focus_subject.name}\n"
            f"类型: {focus_subject.subject_type}\n"
            f"角色: {focus_subject.role}\n"
            f"选择理由: {focus_subject.reason}\n"
            "请让若干条查询显式包含该焦点人物的姓名或职务，以便收集与其决策/承压相关的报道。"
        )
        result = await self.llm.query_json(
            prompts.SEARCH_PLAN_PROMPT, plan_input,
        )
        return result.get("queries", [])

    async def _select_focus_subject(
        self,
        parsed: dict,
        focus_person: str | None,
    ) -> FocusSubject:
        if focus_person and focus_person.strip():
            name = focus_person.strip()
            return FocusSubject(
                name=name,
                subject_type="person",
                role="手动指定",
                reason="用户通过 --focus-person 传入",
                confidence=1.0,
            )
        # 信源池尚未采集：仅基于解构结果自动选择锚点
        decon_text = (
            f"核心事件: {parsed.get('core_event', '')}\n"
            f"日期: {parsed.get('date', '未知')}\n"
            f"人物: {parsed.get('entities', {}).get('people', [])}\n"
            f"组织: {parsed.get('entities', {}).get('organizations', [])}\n"
            f"地点: {parsed.get('entities', {}).get('locations', [])}\n"
            f"关键声明:\n"
            + "\n".join(f"- {c}" for c in parsed.get("key_claims", []))
            + "\n\n说明: 尚未采集信源池，请仅基于以上解构选择视角锚点。"
        )
        try:
            result = await self.llm.query_json(
                prompts.FOCUS_SUBJECT_PROMPT,
                decon_text,
                web_search=False,
            )
            fs = result.get("focus_subject", {})
            return FocusSubject(
                name=fs.get("name", "未知"),
                subject_type=fs.get("subject_type", "person"),
                role=fs.get("role", ""),
                reason=fs.get("reason", ""),
                confidence=float(fs.get("confidence", 0.5)),
            )
        except Exception as e:
            self.console.print(f"  [red]自动选择焦点失败: {e}，使用启发式回退[/red]")
            people = parsed.get("entities", {}).get("people", [])
            orgs = parsed.get("entities", {}).get("organizations", [])
            if people:
                return FocusSubject(
                    name=str(people[0]),
                    subject_type="person",
                    role="解构列表首位人物",
                    reason="焦点选择失败后的回退",
                    confidence=0.3,
                )
            if orgs:
                return FocusSubject(
                    name=str(orgs[0]),
                    subject_type="organization",
                    role="解构列表首位组织",
                    reason="焦点选择失败后的回退",
                    confidence=0.25,
                )
            return FocusSubject(
                name="事件相关方",
                subject_type="person",
                role="未知",
                reason="无可用实体时的占位",
                confidence=0.1,
            )

    async def _build_relative_sequence(
        self,
        focus_subject: FocusSubject,
        pool: list[SourceEntry],
        causal_tree: EventNode,
        event_timeline: list[TimelineEvent],
        history_steps: int,
    ) -> tuple[list[RelativeStep], list[str]]:
        """返回按 index 升序的 E[-N]..E[0]；附带序列构建缺口说明。"""
        gaps: list[str] = []
        timeline_text = "\n".join(
            f"- [{e.significance}] {e.date}: {e.title} — {e.description}"
            for e in event_timeline
        )
        pool_text = "\n".join(
            f"- [{s.outlet}] ({s.source_type}) {s.url}\n"
            + "\n".join(f"    · {f.claim}" for f in s.key_facts[:5])
            for s in pool[:40]
        )
        tree_text = self._tree_to_text(causal_tree)
        user_payload = (
            f"history_steps（N）= {history_steps}\n"
            f"焦点人物: {focus_subject.name} ({focus_subject.subject_type})\n"
            f"角色: {focus_subject.role}\n\n"
            f"=== 事件时间线 ===\n{timeline_text or '(空)'}\n\n"
            f"=== 因果事件树 ===\n{tree_text}\n\n"
            f"=== 信源池（摘录）===\n{pool_text or '(空)'}\n"
        )
        steps: list[RelativeStep] = []
        try:
            result = await self.llm.query_json(
                prompts.RELATIVE_SEQUENCE_PROMPT,
                user_payload,
                web_search=False,
            )
            gaps.extend(result.get("information_gaps", []) or [])
            for s in result.get("steps", []):
                raw_sources = s.get("sources", []) or []
                sources: list[dict] = []
                for src in raw_sources:
                    if isinstance(src, dict):
                        sources.append(src)
                    elif isinstance(src, str):
                        sources.append({"url": "", "title": src})
                steps.append(RelativeStep(
                    index=int(s.get("index", 0)),
                    date=str(s.get("date", "未知")),
                    title=str(s.get("title", "")),
                    description=str(s.get("description", "")),
                    person_view=str(s.get("person_view", "")),
                    actor_goal=str(s.get("actor_goal", "")),
                    constraint=str(s.get("constraint", "")),
                    action_signal=str(s.get("action_signal", "")),
                    sources=sources,
                    verified=bool(s.get("verified", False)),
                    grounded=bool(s.get("grounded", False)),
                    aggregated=bool(s.get("aggregated", False)),
                ))
        except Exception as e:
            self.console.print(f"  [red]相对序列构建失败: {e}[/red]")
            gaps.append(f"LLM 序列构建异常: {e}")

        by_index: dict[int, RelativeStep] = {}
        for st in steps:
            if st.index not in by_index:
                by_index[st.index] = st
        expected = set(range(-history_steps, 1))
        missing = sorted(expected - set(by_index.keys()))
        for idx in missing:
            gaps.append(f"缺失索引 {idx}，已用聚合补位节点填充")
            by_index[idx] = RelativeStep(
                index=idx,
                date="未知",
                title=f"背景补位 (E[{idx}])",
                description="模型未返回该索引；程序聚合补位，请谨慎解读。",
                person_view=f"对 {focus_subject.name} 而言，此阶段为叙事链上的占位补全。",
                actor_goal="",
                constraint="",
                action_signal="",
                sources=[],
                verified=False,
                grounded=False,
                aggregated=True,
            )
        ordered = [by_index[i] for i in range(-history_steps, 1)]
        return ordered, gaps

    def _format_visible_steps_for_predict(
        self, visible: list[RelativeStep],
    ) -> str:
        lines = []
        for s in visible:
            lines.append(
                f"E[{s.index}] {s.date} | {s.title}\n"
                f"  描述: {s.description}\n"
                f"  人物视角: {s.person_view}\n"
                f"  目标/约束/信号: {s.actor_goal} | {s.constraint} | {s.action_signal}"
            )
        return "\n\n".join(lines) if lines else "(无已解锁步骤)"

    async def _predict_step(
        self,
        focus_subject: FocusSubject,
        visible_truth_steps: list[RelativeStep],
        target_index: int,
    ) -> StepPrediction:
        user_in = (
            f"target_index = {target_index}\n\n"
            f"=== 焦点 ===\n"
            f"姓名: {focus_subject.name}\n"
            f"类型: {focus_subject.subject_type}\n"
            f"角色: {focus_subject.role}\n\n"
            f"=== 已解锁真实步骤（仅此部分可见）===\n"
            f"{self._format_visible_steps_for_predict(visible_truth_steps)}\n"
        )
        result = await self.llm.query_json(
            prompts.STEP_PREDICT_PROMPT,
            user_in,
            web_search=False,
        )
        p = result.get("prediction", {})
        return StepPrediction(
            target_index=int(p.get("target_index", target_index)),
            predicted_title=str(p.get("predicted_title", "")),
            predicted_description=str(p.get("predicted_description", "")),
            predicted_person_view=str(p.get("predicted_person_view", "")),
            rationale=str(p.get("rationale", "")),
            confidence=float(p.get("confidence", 0.0)),
        )

    async def _compare_step(
        self,
        prediction: StepPrediction,
        truth_step: RelativeStep,
    ) -> dict:
        pred_text = (
            f"预测 E[{prediction.target_index}]:\n"
            f"  title: {prediction.predicted_title}\n"
            f"  description: {prediction.predicted_description}\n"
            f"  person_view: {prediction.predicted_person_view}\n"
            f"  rationale: {prediction.rationale}\n"
        )
        truth_text = (
            f"真值 E[{truth_step.index}]:\n"
            f"  title: {truth_step.title}\n"
            f"  description: {truth_step.description}\n"
            f"  person_view: {truth_step.person_view}\n"
            f"  date: {truth_step.date}\n"
        )
        result = await self.llm.query_json(
            prompts.STEP_COMPARE_PROMPT,
            f"{pred_text}\n{truth_text}",
            web_search=False,
        )
        return {
            "matched": bool(result.get("matched", False)),
            "score": float(result.get("score", 0.0)),
            "note": str(result.get("note", "")),
        }

    async def _run_stepwise_replay(
        self,
        truth_sequence: list[RelativeStep],
        focus_subject: FocusSubject,
        history_steps: int,
        strict_replay: bool,
        continue_on_mismatch: bool,
    ) -> tuple[list[StepPrediction], int | None, float]:
        """从 E[-N] 起仅解锁首步，逐点预测 E[-N+1]..E[0] 并与真值比对。"""
        predictions: list[StepPrediction] = []
        stopped_at: int | None = None
        index_map = {s.index: s for s in truth_sequence}
        min_idx = -history_steps
        if min_idx not in index_map:
            self.console.print(f"  [red]truth_sequence 缺少 E[{min_idx}][/red]")
            return predictions, None, 0.0

        visible: list[RelativeStep] = [index_map[min_idx]]

        for k in range(min_idx + 1, 1):
            truth_k = index_map.get(k)
            if truth_k is None:
                self.console.print(f"  [red]truth_sequence 缺少 E[{k}][/red]")
                break
            pred = await self._predict_step(focus_subject, visible, k)
            cmp_result = await self._compare_step(pred, truth_k)
            pred.matched = cmp_result["matched"]
            pred.match_score = cmp_result["score"]
            pred.match_note = cmp_result["note"]
            pred.truth_step = truth_k
            predictions.append(pred)

            if pred.matched or continue_on_mismatch:
                visible.append(truth_k)
            elif not strict_replay:
                # 非严格：仍回填真值以便继续评估后续点
                visible.append(truth_k)
            else:
                stopped_at = k
                break

        n = len(predictions)
        accuracy = (
            sum(1 for p in predictions if p.matched) / n if n else 0.0
        )
        return predictions, stopped_at, accuracy

    async def _forecast_next_step(
        self,
        truth_sequence: list[RelativeStep],
        predictions: list[StepPrediction],
        focus_subject: FocusSubject,
    ) -> tuple[list[ForecastScenario], str]:
        truth_lines = []
        for s in truth_sequence:
            truth_lines.append(
                f"E[{s.index}] {s.date} | {s.title}\n"
                f"  {s.description}\n"
                f"  视角: {s.person_view}\n"
                f"  聚合补位: {s.aggregated} | 有据: {s.grounded}"
            )
        pred_lines = []
        for p in predictions:
            pred_lines.append(
                f"目标 E[{p.target_index}] matched={p.matched} score={p.match_score:.2f}\n"
                f"  预测标题: {p.predicted_title}\n"
                f"  真值标题: {(p.truth_step.title if p.truth_step else '?')}\n"
                f"  note: {p.match_note}"
            )
        user_in = (
            f"焦点: {focus_subject.name} ({focus_subject.subject_type}) — {focus_subject.role}\n\n"
            f"=== 完整真实序列 ===\n" + "\n".join(truth_lines) + "\n\n"
            f"=== Replay 逐步预测记录 ===\n" + "\n".join(pred_lines)
        )
        try:
            result = await self.llm.query_json(
                prompts.NEXT_STEP_FORECAST_PROMPT,
                user_in,
                web_search=False,
            )
        except Exception as e:
            self.console.print(f"  [red]+1 场景预测失败: {e}[/red]")
            return [], str(e)

        scenarios: list[ForecastScenario] = []
        for sc in (result.get("scenarios") or [])[:3]:
            scenarios.append(ForecastScenario(
                title=str(sc.get("title", "")),
                description=str(sc.get("description", "")),
                why=str(sc.get("why", "")),
                triggers=list(sc.get("triggers", []) or []),
                invalidators=list(sc.get("invalidators", []) or []),
                confidence=float(sc.get("confidence", 0.0)),
            ))
        summary = str(result.get("summary", ""))
        return scenarios, summary

    # ══ Phase 1: 解构 + 搜索规划 ═══════════════════════════

    async def _deconstruct(self, news_input: str) -> dict:
        return await self.llm.query_json(
            prompts.DECONSTRUCT_PROMPT, news_input,
        )

    async def _plan_searches(self, parsed: dict) -> list[dict]:
        plan_input = (
            f"核心事件: {parsed.get('core_event', '')}\n"
            f"日期: {parsed.get('date', '未知')}\n"
            f"人物: {parsed.get('entities', {}).get('people', [])}\n"
            f"组织: {parsed.get('entities', {}).get('organizations', [])}\n"
            f"地点: {parsed.get('entities', {}).get('locations', [])}\n"
            f"关键声明:\n"
            + "\n".join(f"- {c}" for c in parsed.get("key_claims", []))
        )
        result = await self.llm.query_json(
            prompts.SEARCH_PLAN_PROMPT, plan_input,
        )
        return result.get("queries", [])

    # ══ Track A: 多轮信源采集 ══════════════════════════════

    async def _trace_sources(
        self, queries: list[dict],
    ) -> tuple[list[SourceEntry], dict]:
        # A1: 并行多角度搜索
        self.console.print("  [yellow]Track A[/yellow]: 并行搜索中…")
        tasks = [self._search_one_angle(q) for q in queries]
        raw_batches = await asyncio.gather(*tasks)

        # A2: 合并去重
        pool = self._merge_sources(raw_batches)
        self.console.print(
            f"  [yellow]Track A[/yellow]: "
            f"合并去重后 {len(pool)} 个信源"
        )

        # A3: 交叉验证
        self.console.print("  [yellow]Track A[/yellow]: 交叉验证中…")
        verify_input = self._format_pool_for_verify(pool)
        verify_data = await self.llm.query_json(
            prompts.SOURCE_VERIFY_PROMPT,
            verify_input,
            web_search=True,
        )
        self.console.print("  [yellow]Track A[/yellow]: 信源考古完成 ✓")
        return pool, verify_data

    async def _search_one_angle(self, q: dict) -> list[dict]:
        angle = q.get("angle", "")
        query = q.get("query", "")
        self.console.print(
            f"    [yellow]搜索[/yellow] [{angle}] {query}"
        )
        try:
            result = await self.llm.query_json(
                prompts.SOURCE_TRACE_PROMPT,
                f"搜索主题: {query}\n搜索角度: {angle}",
                web_search=True,
            )
            return result.get("sources", [])
        except Exception as e:
            self.console.print(f"    [red]搜索失败: {e}[/red]")
            return []

    @staticmethod
    def _merge_sources(raw_batches: list[list[dict]]) -> list[SourceEntry]:
        seen_urls: set[str] = set()
        seen_outlets: set[str] = set()
        pool: list[SourceEntry] = []

        for batch in raw_batches:
            for s in batch:
                url = s.get("url", "").strip()
                outlet = s.get("outlet", "").strip()

                # URL 去重
                if url and url in seen_urls:
                    continue
                # 无 URL 时按媒体名去重
                if not url and outlet and outlet in seen_outlets:
                    continue

                if url:
                    seen_urls.add(url)
                if outlet:
                    seen_outlets.add(outlet)

                facts = [
                    FactClaim(
                        claim=f.get("claim", ""),
                        source_urls=[url] if url else [],
                    )
                    for f in s.get("facts", [])
                ]

                pool.append(SourceEntry(
                    outlet=outlet,
                    url=url,
                    publish_time=s.get("publish_time", ""),
                    source_type=s.get("source_type", ""),
                    is_original=s.get("is_original", False),
                    key_facts=facts,
                ))

        return pool

    @staticmethod
    def _format_pool_for_verify(pool: list[SourceEntry]) -> str:
        lines = []
        for s in pool:
            header = f"[{s.outlet}] ({s.source_type}) {s.url}"
            facts = "\n".join(
                f"  - {f.claim}" for f in s.key_facts
            )
            lines.append(f"{header}\n{facts}")
        return "以下是多个信源的报道及其事实声明，请逐条交叉验证:\n\n" + "\n\n".join(lines)

    # ══ Track B: 有据因果挖掘 ══════════════════════════════

    async def _dig_causes(self, event: EventNode, depth: int) -> EventNode:
        if depth >= self.max_depth:
            return event

        indent = "  " + "  " * depth
        self.console.print(
            f"{indent}[magenta]Track B[/magenta]: "
            f"挖掘因果 (层 {depth + 1}/{self.max_depth}) — {event.title}"
        )

        prompt_input = (
            f"事件: {event.title}\n"
            f"日期: {event.date}\n"
            f"概要: {event.summary}"
        )

        try:
            result = await self.llm.query_json(
                prompts.CAUSAL_DIG_PROMPT,
                prompt_input,
                web_search=True,
            )
        except Exception as e:
            self.console.print(f"{indent}[red]因果挖掘失败: {e}[/red]")
            return event

        children: list[EventNode] = []
        for c in result.get("causes", []):
            raw_sources = c.get("sources", [])
            sources = []
            for src in raw_sources:
                if isinstance(src, dict):
                    sources.append(src)
                elif isinstance(src, str):
                    sources.append({"url": "", "title": src})

            child = EventNode(
                title=c.get("title", "未知"),
                date=c.get("date", "未知"),
                summary=c.get("summary", ""),
                sources=sources,
                relation=c.get("relation", ""),
                confidence=c.get("confidence", 0.5),
                grounded=c.get("grounded", False),
            )
            if not c.get("is_root", False):
                child = await self._dig_causes(child, depth + 1)
            children.append(child)

        event.causes = children

        if depth == 0:
            self.console.print("  [magenta]Track B[/magenta]: 因果挖掘完成 ✓")

        return event

    # ══ Phase 3: 锚定验证 + 综合研判 ══════════════════════

    async def _ground_check(
        self, tree: EventNode, pool: list[SourceEntry],
    ) -> EventNode:
        self.console.print("  锚定检查: 因果树 vs 信源池…")

        tree_text = self._tree_to_text(tree)
        pool_text = "\n".join(
            f"- [{s.outlet}] {s.url}: "
            + "; ".join(f.claim for f in s.key_facts)
            for s in pool
        )

        try:
            result = await self.llm.query_json(
                prompts.GROUNDING_CHECK_PROMPT,
                f"=== 因果事件树 ===\n{tree_text}\n\n"
                f"=== 信源池 ===\n{pool_text}",
            )
        except Exception as e:
            self.console.print(f"  [red]锚定检查失败: {e}[/red]")
            return tree

        check_map = {
            c["event_title"]: c
            for c in result.get("checks", [])
        }
        self._apply_grounding(tree, check_map)
        return tree

    def _apply_grounding(self, node: EventNode, check_map: dict) -> None:
        check = check_map.get(node.title)
        if check:
            node.grounded = check.get("grounded", node.grounded)
        for child in node.causes:
            self._apply_grounding(child, check_map)

    @staticmethod
    def _count_grounding(tree: EventNode) -> tuple[int, int]:
        total, grounded = 0, 0

        def walk(node: EventNode) -> None:
            nonlocal total, grounded
            for child in node.causes:
                total += 1
                if child.grounded:
                    grounded += 1
                walk(child)

        walk(tree)
        return total, grounded

    @staticmethod
    def _compute_credibility(
        pool: list[SourceEntry],
        verify_data: dict,
        causal_tree: EventNode,
    ) -> CredibilityBreakdown:
        source_types = sorted({s.source_type for s in pool if s.source_type})
        n_consistent = len(verify_data.get("consistent_facts", []))
        n_disputed = len(verify_data.get("disputed_facts", []))
        total, grounded = NewsTracingAgent._count_grounding(causal_tree)
        return CredibilityBreakdown(
            source_count=len(pool),
            source_types=source_types,
            consistent_count=n_consistent,
            disputed_count=n_disputed,
            grounding_rate=grounded / total if total else 0.0,
            grounded_count=grounded,
            total_causal_nodes=total,
        )

    # ── 时间线构建 ─────────────────────────────────────────

    async def _build_timeline(
        self,
        pool: list[SourceEntry],
        causal_tree: EventNode,
    ) -> list[TimelineEvent]:
        self.console.print("  构建事件时间线…")

        facts_text = self._collect_facts_text(pool)
        causal_text = self._flatten_tree_text(causal_tree)

        build_input = (
            f"=== 信源事实 ===\n{facts_text}\n\n"
            f"=== 因果事件 ===\n{causal_text}"
        )

        try:
            result = await self.llm.query_json(
                prompts.TIMELINE_BUILD_PROMPT, build_input,
            )
        except Exception as e:
            self.console.print(f"  [red]时间线构建失败: {e}[/red]")
            return []

        events: list[TimelineEvent] = []
        for ev in result.get("events", []):
            raw_sources = ev.get("sources", [])
            sources = []
            for src in raw_sources:
                if isinstance(src, dict):
                    sources.append(src)
                elif isinstance(src, str):
                    sources.append({"url": "", "title": src})

            src_count = ev.get("source_count", len(sources))
            events.append(TimelineEvent(
                date=ev.get("date", "未知"),
                title=ev.get("title", ""),
                description=ev.get("description", ""),
                significance=ev.get("significance", "重要"),
                sources=sources,
                verified=src_count >= 2,
                causal_links=ev.get("causal_links", []),
            ))

        return events

    @staticmethod
    def _collect_facts_text(pool: list[SourceEntry]) -> str:
        lines: list[str] = []
        for s in pool:
            for f in s.key_facts:
                lines.append(
                    f"- [{s.outlet}] {f.claim} (来源: {s.url or '无链接'})"
                )
        return "\n".join(lines) if lines else "(无事实数据)"

    @staticmethod
    def _flatten_tree_text(node: EventNode) -> str:
        items: list[str] = []

        def walk(n: EventNode) -> None:
            src_info = ", ".join(
                s.get("title", s.get("url", "?")) for s in n.sources[:3]
            )
            items.append(
                f"- {n.title} ({n.date}): {n.summary}"
                + (f" [来源: {src_info}]" if src_info else "")
            )
            for child in n.causes:
                walk(child)

        walk(node)
        return "\n".join(items) if items else "(无因果数据)"

    # ── 多方视角对比 ───────────────────────────────────────

    async def _analyze_perspectives(
        self,
        timeline: list[TimelineEvent],
        pool: list[SourceEntry],
    ) -> list[TimelineEvent]:
        major_events = [e for e in timeline if e.significance == "重大"]
        if not major_events:
            return timeline

        self.console.print(
            f"  多方视角分析 ({len(major_events)} 个重大事件)…"
        )

        events_text = "\n".join(
            f"- {e.title}: {e.description}" for e in major_events
        )
        pool_text = "\n".join(
            f"- [{s.outlet}] ({s.source_type}): "
            + "; ".join(f.claim for f in s.key_facts[:3])
            for s in pool
        )

        try:
            result = await self.llm.query_json(
                prompts.PERSPECTIVE_PROMPT,
                f"=== 重大事件 ===\n{events_text}\n\n"
                f"=== 信源池 ===\n{pool_text}",
            )
        except Exception as e:
            self.console.print(f"  [red]视角分析失败: {e}[/red]")
            return timeline

        perspective_map: dict[str, list[dict]] = {}
        for p in result.get("perspectives", []):
            perspective_map[p.get("event_title", "")] = p.get("views", [])

        for ev in timeline:
            views = perspective_map.get(ev.title, [])
            if views:
                ev.perspectives = views

        self.console.print("  多方视角分析完成 ✓")
        return timeline

    # ── 直答生成 ───────────────────────────────────────────

    async def _generate_direct_response(
        self,
        news_input: str,
        pool: list[SourceEntry],
        verify_data: dict,
        event_timeline: list[TimelineEvent],
        causal_tree: EventNode,
        cred: CredibilityBreakdown,
    ) -> str:
        self.console.print("  生成直答…")

        timeline_text = "\n".join(
            f"- [{e.significance}] {e.date}: {e.title} — {e.description}"
            for e in event_timeline[:15]
        )
        tree_text = self._tree_to_text(causal_tree)

        dr_input = (
            f"=== 用户原始输入 ===\n{news_input}\n\n"
            f"=== 事件时间线 ({len(event_timeline)} 个事件) ===\n"
            f"{timeline_text}\n\n"
            f"=== 交叉验证 ===\n"
            f"一致事实: {verify_data.get('consistent_facts', [])}\n"
            f"分歧事实: {verify_data.get('disputed_facts', [])}\n\n"
            f"=== 因果链摘要 ===\n{tree_text}\n\n"
            f"=== 可信度原始数据 ===\n"
            f"信源总数: {cred.source_count}\n"
            f"信源类型: {', '.join(cred.source_types)}\n"
            f"多源一致事实: {cred.consistent_count} 条\n"
            f"存在分歧事实: {cred.disputed_count} 条\n"
            f"因果锚定: {cred.grounded_count}/{cred.total_causal_nodes}"
        )

        try:
            return await self.llm.query(
                prompts.DIRECT_RESPONSE_PROMPT, dr_input,
            )
        except Exception as e:
            self.console.print(f"  [red]直答生成失败: {e}[/red]")
            return ""

    # ── 综合研判 ───────────────────────────────────────────

    async def _synthesize(
        self,
        pool: list[SourceEntry],
        verify_data: dict,
        causal_tree: EventNode,
        event_timeline: list[TimelineEvent],
        cred: CredibilityBreakdown,
        direct_response: str,
    ) -> TracingReport:
        self.console.print("  生成详细研判…")

        timeline_text = "\n".join(
            f"- [{e.significance}] {e.date}: {e.title} — {e.description}"
            for e in event_timeline
        )
        pool_summary = "\n".join(
            f"- [{s.outlet}] ({s.source_type}) {s.publish_time}"
            for s in pool
        )
        tree_text = self._tree_to_text(causal_tree)

        synthesis_input = (
            f"=== 事件时间线 ({len(event_timeline)} 个事件) ===\n"
            f"{timeline_text}\n\n"
            f"=== 信源池 ({len(pool)} 个来源) ===\n{pool_summary}\n\n"
            f"=== 交叉验证 ===\n"
            f"一致事实: {verify_data.get('consistent_facts', [])}\n"
            f"分歧事实: {verify_data.get('disputed_facts', [])}\n\n"
            f"=== 因果事件树 ===\n{tree_text}\n\n"
            f"=== 因果锚定 ===\n"
            f"{cred.grounded_count}/{cred.total_causal_nodes}"
        )

        result = await self.llm.query_json(
            prompts.SYNTHESIS_PROMPT, synthesis_input,
        )

        source_timeline = sorted(pool, key=lambda s: s.publish_time or "")

        consistent_facts = [
            FactClaim(
                claim=f.get("claim", f) if isinstance(f, dict) else str(f),
                source_urls=f.get("source_urls", []) if isinstance(f, dict) else [],
                agreed=True,
            )
            for f in verify_data.get("consistent_facts", [])
        ]
        disputed_facts = [
            FactClaim(
                claim=f.get("claim", f) if isinstance(f, dict) else str(f),
                source_urls=f.get("source_urls", []) if isinstance(f, dict) else [],
                agreed=False,
            )
            for f in verify_data.get("disputed_facts", [])
        ]

        return TracingReport(
            event=causal_tree,
            direct_response=direct_response,
            credibility=cred,
            event_timeline=event_timeline,
            sources=pool,
            source_timeline=source_timeline,
            key_findings=result.get("key_findings", []),
            information_gaps=result.get("information_gaps", []),
            consistent_facts=consistent_facts,
            disputed_facts=disputed_facts,
            causal_summary=result.get("causal_summary", ""),
            bias_notes=result.get("bias_notes", []),
        )

    # ══ 辅助 ═══════════════════════════════════════════════

    @staticmethod
    def _tree_to_text(node: EventNode, indent: int = 0) -> str:
        prefix = "  " * indent
        rel = f" [{node.relation}]" if node.relation else ""
        grounded_tag = " [有据]" if node.grounded else ""
        lines = [
            f"{prefix}- {node.title} ({node.date}){rel}{grounded_tag}",
            f"{prefix}  {node.summary}",
        ]
        for child in node.causes:
            lines.append(NewsTracingAgent._tree_to_text(child, indent + 1))
        return "\n".join(lines)
