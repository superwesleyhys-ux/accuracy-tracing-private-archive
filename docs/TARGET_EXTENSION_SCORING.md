# Target-extension offline comparison

`experiments/target_extension_compare_score.py` compares the original agent, the
staged ψ pipeline, and the target-extension pipeline from retained artifacts. It
does not import an inference client, read an API credential, or make a network
request.

The extension run defines the case denominator. The scorer requires those exact
target and material contracts to exist in both baseline runs, then recalculates
all three arms on that same subset. A failed execution remains a failed scheduled
task; it is never relabeled `unverifiable`.

```bash
python experiments/target_extension_compare_score.py \
  --gold experiments/proof_pilot/gold.json \
  --original-run reports/staged-baseline-01 \
  --staged-run reports/staged-dev-04 \
  --extension-run reports/target-extension-dev-01 \
  --output reports/target-extension-comparison-01.json
```

The output separates four questions:

| Section | What it measures |
|---|---|
| `arms` and `label_comparisons` | Completion, fixed-denominator task success, conditional label agreement, pairwise fixes and breaks |
| `round1_to_final` | First-round to final label fixes and breaks among completed cases with a real first checkpoint |
| `origins` / `edges` inside each arm | Corpus-relative documentary-root and direct-edge accuracy on explicitly evaluable cases |
| `target_plan_probe_coverage` | Checksummed required-probe presence, deterministic stage routing, delivery to an executed stage, and accepted judgement-review exposure |
| `loop_audit` | Target-plan, material-critic, and judgement-critic calls, repair requests, and bounded repair attempts |
| `usage_total` and deltas | Calls, input/output/reasoning/visible-output tokens, and summed service elapsed seconds |

Probe coverage is deliberately not folded into label accuracy. A value of `1.0`
means the retained checklist was complete and reached every routed stage; it does
not show that the model answered each probe correctly. Provider-reported reasoning
tokens are likewise a resource diagnostic, not a quality score.

The current eight cases are previously seen development examples with provisional
AI-reviewed references. They can expose regressions and accounting errors, but
cannot prove a real-world accuracy improvement. That claim requires a separately
frozen held-out set with independent adjudication.
