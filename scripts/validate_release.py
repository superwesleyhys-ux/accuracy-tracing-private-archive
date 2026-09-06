"""Reproduce the current offline release checks without rewriting old reports."""
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newsverify import __version__
from newsverify.comparison import compare
from newsverify.evaluation import evaluate
from newsverify.trace_demo import run_demo


SERIES = ".".join(__version__.split(".")[:2])
TAG = "v" + SERIES


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def _run(command, *, cwd=ROOT):
    return subprocess.run(
        [str(item) for item in command], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def package_smoke():
    """Build the real wheel, install it in isolation, and test source-only stages."""
    result = {
        "status": "failed",
        "version": __version__,
        "distribution_scope": "core CLI/trace engine in wheel; staged experiments from source checkout",
    }
    with tempfile.TemporaryDirectory(prefix="newsverify-release-") as temporary:
        temporary_path = Path(temporary)
        wheelhouse = temporary_path / "wheelhouse"
        wheelhouse.mkdir()
        built = _run([
            sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
            "--disable-pip-version-check", "--wheel-dir", wheelhouse,
        ])
        if built.returncode:
            result["failure_stage"] = "wheel_build"
            result["output_tail"] = built.stdout[-2000:]
            return result
        wheels = sorted(wheelhouse.glob(
            f"newsverify_harness-{__version__}-*.whl"))
        if len(wheels) != 1:
            result["failure_stage"] = "wheel_selection"
            result["wheel_count"] = len(wheels)
            return result
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        required = {
            "newsverify/__init__.py", "newsverify/cli.py",
            "newsverify/decisions.py", "newsverify/provenance.py",
            "newsverify/retrieval.py",
        }
        missing = sorted(required - names)
        if missing:
            result["failure_stage"] = "wheel_contents"
            result["missing"] = missing
            return result

        environment = temporary_path / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = (environment / "Scripts" / "python.exe"
                  if sys.platform == "win32" else environment / "bin" / "python")
        installed = _run([
            python, "-m", "pip", "install", "--no-deps",
            "--disable-pip-version-check", wheel,
        ], cwd=temporary_path)
        if installed.returncode:
            result["failure_stage"] = "wheel_install"
            result["output_tail"] = installed.stdout[-2000:]
            return result
        imported = _run([
            python, "-c",
            "import newsverify; assert newsverify.__version__ == " + repr(__version__),
        ], cwd=temporary_path)
        demo = _run([python, "-m", "newsverify", "demo"], cwd=temporary_path)
        source_staged = _run([
            sys.executable, "experiments/loop_compare.py", "--help",
        ])
        if imported.returncode or demo.returncode or source_staged.returncode:
            result["failure_stage"] = "installed_or_source_smoke"
            result["output_tail"] = (
                imported.stdout + demo.stdout + source_staged.stdout)[-2000:]
            return result
        result.update({
            "status": "passed",
            "wheel": wheel.name,
            "wheel_required_files": sorted(required),
            "installed_version": __version__,
            "installed_core_demo": "passed",
            "source_staged_help": "passed",
        })
    return result


def main():
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    groups = Counter(test.id().split(".")[0] for test in flatten(suite))
    transcript = io.StringIO()
    test_result = unittest.TextTestRunner(
        stream=transcript, verbosity=2).run(suite)
    (reports / f"test-output-{TAG}.txt").write_text(
        transcript.getvalue(), encoding="utf-8")

    read = lambda name: json.loads(
        (ROOT / "examples" / name).read_text(encoding="utf-8"))
    trace = run_demo()
    metrics = evaluate(
        read("evaluation_gold.json"), read("evaluation_predictions.json"))
    comparison = compare(
        read("evaluation_gold.json"), read("comparison_baseline.json"),
        read("comparison_candidate.json"), bootstrap_samples=100, seed=0)
    for name, payload in (
            (f"trace-demo-{TAG}.json", trace),
            (f"all-metrics-{TAG}.json", metrics),
            (f"comparison-{TAG}.json", comparison)):
        (reports / name).write_text(json.dumps(
            payload, ensure_ascii=False, indent=2,
            allow_nan=False) + "\n", encoding="utf-8")

    packaging = package_smoke()
    (reports / f"package-smoke-{TAG}.json").write_text(
        json.dumps(packaging, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    validation = {
        "release": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tests_run": test_result.testsRun,
        "failures": len(test_result.failures),
        "errors": len(test_result.errors),
        "skipped": len(test_result.skipped),
        "successful": test_result.wasSuccessful(),
        "test_groups": dict(groups),
        "metric_contracts_checked": ["VP", "FR", "TR", "SR", "EN", "CA", "HFAR"],
        "trace_demo_usage": trace["usage"],
        "trace_demo_provenance": trace["provenance_status"],
        "trace_demo_fact_status": trace["fact_status"],
        "trace_demo_stop": trace["stop_reason"],
        "packaging": packaging,
        "all_data_synthetic": True,
        "live_news_requests": 0,
        "model_api_calls": 0,
        "staged_model_run_performed": False,
        "realworld_accuracy_measured": False,
    }
    (reports / f"validation-{TAG}.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    lines = [
        f"# Accuracy Tracing {TAG}：离线验收记录", "",
        f"生成时间：{validation['generated_at']}", "",
        "本报告只描述当前源码的离线测试和安装烟测。没有调用新闻检索服务或模型 API，没有测得真实新闻准确率。", "",
        "## 运行结果", "",
        f"- 测试：{test_result.testsRun} 项；失败 {len(test_result.failures)}；错误 {len(test_result.errors)}；跳过 {len(test_result.skipped)}。",
        f"- 核心 wheel 安装检查：{packaging['status']}。",
        "- staged 分阶段运行时：源码 checkout 导入/CLI 帮助烟测；本次没有执行模型推理。", "",
        "| 测试组 | 数量 |", "|---|---:|",
    ]
    lines += [f"| {name} | {count} |" for name, count in sorted(groups.items())]
    lines += [
        "", "## 双回环演示", "",
        f"- 检索轮次：{trace['usage']['rounds']}。",
        f"- 不同材料版本：{trace['usage']['unique_versions']}。",
        f"- 分解调用：{trace['usage']['decomposition_calls']}。",
        f"- 验证调用：{trace['usage']['verification_calls']}。",
        f"- 演示状态：`{trace['provenance_status']}` / `{trace['fact_status']}` / `{trace['stop_reason']}`。",
        "", "演示使用程序化 fixture 标注，只验证路由、顺序、预算、状态与审计合同。它不是通用语义模型实测，也不证明拆分提示词提升了准确率。",
        "", "## 指标算术样例", "",
        "以下分数属于四条故意含错的手写预测，只检查评分器，不代表项目或任何模型的性能。", "",
        "| 指标 | 手写样例计算值 |", "|---|---:|",
    ]
    lines += [
        f"| {key} | {metrics['metrics'][key]['value']:.6f} |"
        for key in validation["metric_contracts_checked"]
    ]
    lines += [
        "", "## 仍需外部验证", "",
        "- 真实新闻检索适配器、来源信任与历史可用性审计。",
        "- 独立人工 gold、隐藏且按事件/时间隔离的真实新闻测试集。",
        "- 等总预算 staged/monolithic 消融；当前离线通过不能推导真实准确率增益。",
        "- 生产并发、网络故障、成本和长期稳定性测试。", "",
    ]
    (reports / f"VALIDATION_{TAG.upper()}.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if test_result.wasSuccessful() and packaging["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
