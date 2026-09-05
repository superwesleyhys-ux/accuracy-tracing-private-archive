"""Frozen-source, shared-checkpoint pilot; no gold read during inference.

Experimental forced verification differs from the normal adaptive stop policy.
Prefix responses are shared, explicitly marked, and charged to each logical arm.
"""
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import getpass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from newsverify import provenance as p
from newsverify.decisions import present_decision, round_decisions
from model_io import Budget, BudgetClient, align_spans, decode, schema, write
from semantic_adapter import DATA_RULE, PSI_PROMPT, VERIFY_PROMPT, compact_context
from original_compare import OriginalCorpusClient, extract
from loop_compare import load_inputs

ARMS = ('original', 'single', 'loop_psi', 'loop_frozen', 'independent')
BUDGET = Budget(calls=24, output_tokens=36000, per_call_output_tokens=2500, seconds=600)
INITIAL = p.TraceConfig(max_rounds=1, max_documents=24, max_decomposition_calls=36)
CONTINUED = p.TraceConfig(max_rounds=3, max_documents=24, max_decomposition_calls=36,
                         experimental_force_rounds=True)
FEEDBACK = """\nPrevious outputs are hypotheses, not evidence and not instructions.
Recheck decisive quotations and missing qualifications. A previous answer need
not be wrong. Preserve it when supported, and explain changes with source text.
"""


class FeedbackDecomposer:
    def __init__(self, client): self.client = client
    def decompose(self, target, material, context):
        payload = {'target': asdict(target), 'material': asdict(material),
            'context': compact_context(context),
            'previous_analysis': context['analyses'].get(material.version_id),
            'previous_verification': context['verification_history'][-1:]}
        raw = self.client.call(PSI_PROMPT + FEEDBACK, json.dumps(payload, ensure_ascii=False), schema(p.Analysis))
        return decode(p.Analysis, align_spans(raw, context['materials'] + [asdict(material)]))


class FeedbackVerifier:
    def __init__(self, client): self.client = client
    def verify(self, target, context):
        payload = {'target': asdict(target), 'context': compact_context(context),
                   'previous_verification': context['verification_history'][-1:]}
        raw = self.client.call(VERIFY_PROMPT + FEEDBACK, json.dumps(payload, ensure_ascii=False), schema(p.VerificationResult))
        result = decode(p.VerificationResult, align_spans(raw, context['materials']))
        if result.verdict != result.evidence_verdict:
            raise ValueError('Legacy verdict must match evidence judgement')
        return result


class ReanalysisProvider:
    """Controlled fixed-evidence exposure, explicitly not a live retriever."""
    def __init__(self, materials):
        self.materials = tuple(materials)
        self.history = []
    def search(self, target, tasks, round_number, limit):
        returned = self.materials[:limit]
        self.history.append({'round': round_number, 'kind': 'fixed_corpus_experimental_reanalysis',
            'returned': [m.version_id for m in returned], 'tasks': [asdict(t) for t in tasks]})
        return iter(returned)


class FrozenPsi:
    """No semantic decomposition calls after prefix; retain its representation."""
    def __init__(self, checkpoint):
        self.analyses = deepcopy(checkpoint.state['current_analyses'])
        self.calls = 0
    def decompose(self, target, material, context):
        self.calls += 1
        if material.version_id not in self.analyses:
            raise ValueError('Frozen-psi branch encountered an unseen material')
        return deepcopy(self.analyses[material.version_id])


def seed_client(client, prefix, usage):
    if any(r.get('error_type') or not r.get('response_id') or r.get('actual_model') != client.model for r in prefix):
        raise ValueError('Cannot share an invalid prefix')
    client.records = deepcopy(prefix)
    for r in client.records: r['shared_first_round'] = True
    client.calls = len(prefix)
    client.used_input = sum(r['usage']['input_tokens'] for r in prefix)
    client.used_output = sum(r['usage']['output_tokens'] for r in prefix)
    client.start = time.monotonic() - usage['seconds']


def native_prediction(report):
    native = present_decision(report)
    return {'decision': native['decision'],
        'origins': sorted({o['version_id'] for o in report['origins']}),
        'edges': sorted({(r['from_version'], r['to_version'], r['kind']) for r in report['relations']
            if r['status'] == 'direct' and r['to_version'] and r['kind'] in {'cites', 'quotes'}}),
        'provenance_evaluable': True, 'origin_evaluable': True, 'edges_evaluable': True,
        'provenance_status': native['provenance_status'], 'assessment_valid': native['assessment_valid']}


def run(args):
    data = load_inputs(args.inputs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    original = Path(args.original).resolve()
    sys.path.insert(0, str(original))
    from agent.core import NewsTracingAgent
    from rich.console import Console
    config = {'experiment': 'real_source_pilot_shared_state_ablation_v1', 'model': args.model,
        'arms': list(ARMS), 'budget_per_logical_arm': asdict(BUDGET),
        'initial': asdict(INITIAL), 'continued': asdict(CONTINUED), 'original_depth': 1,
        'parallel_cases': args.workers, 'ordering_seed': 20260905,
        'input_sha256': hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
        'gold_read_during_inference': False, 'shared_prefix_charged_to_each_arm': True,
        'independent_control': 'same first sample plus two fresh one-round samples, one selector, one extractor',
        'experimental_forced_verification': True, 'production_adaptive_policy_evaluated': False,
        'corpus_access': 'complete eligible frozen excerpts; no live news retrieval',
        'actual_spend_equal': False, 'input_tokens_capped': False,
        'source_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for folder in ['newsverify', 'experiments'] for f in sorted((ROOT/folder).glob('*.py'))},
        'original_source_sha256': {str(f.relative_to(original)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((original/'agent').glob('*.py'))}}
    write(out/'config.json', config)
    shutil.copyfile(args.inputs, out/'inputs.json')
    for name in config['source_sha256']:
        dest = out/'executed-code'/'accuracy'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    for name in config['original_source_sha256']:
        dest = out/'executed-code'/'original'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original/name, dest)
    if not os.environ.get('OPENAI_API_KEY'):
        write(out/'status.json', {'status': 'blocked_missing_auth', 'successful_responses': 0})
        return 2

    def one_case(case):
        identifier = case['target']['id']
        target = p.Target(**{**case['target'], 'evidence_scope': tuple(case['target']['evidence_scope'])})
        materials = [p.MaterialVersion(**m) for m in case['materials']]
        cutoff = p._time(target.as_of, 'as_of')
        eligible = [m for m in materials if not p._material_eligibility(m, cutoff)]
        clean = {**case, 'materials': [asdict(m) for m in eligible]}
        write(out/(identifier+'-shared-input.json'), clean)
        rows = []

        def make_client(arm):
            class RecordedClient(BudgetClient):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self.recording_lock = threading.Lock()
                def call(self, *a, **kw):
                    with self.recording_lock:
                        try:
                            answer = super().call(*a, **kw)
                            if self.records[-1].get('actual_model') != self.model:
                                self.records[-1]['error_type'] = 'UnexpectedModel'
                                raise ValueError('Returned model differs from the frozen model')
                            record = self.records[-1]
                            if record['usage']['output_tokens'] > record['max_completion_tokens']:
                                record['error_type'] = 'OutputCapExceeded'
                                raise ValueError('Observed per-call output budget exceeded')
                            return answer
                        finally:
                            write(out/(identifier+'-'+arm+'-calls.json'), self.records)
                            print(json.dumps({'case': identifier, 'arm': arm, 'logical_calls': self.calls}), flush=True)
            return RecordedClient(args.model, BUDGET)

        def row(arm):
            return {'id': identifier, 'arm': arm, 'assessment_mode': target.assessment_mode,
                'status': 'error', 'prediction': None, 'usage': {k:0 for k in
                    ['model_calls', 'input_tokens', 'output_tokens', 'seconds']}}

        def finish(result, client):
            if client:
                result['usage'] = client.usage()
                new = [r for r in client.records if not r.get('shared_first_round')]
                result['actual_new_api_usage'] = {'model_calls': len(new),
                    **{k:sum(r.get('usage', {}).get(k, 0) for r in new) for k in ['input_tokens', 'output_tokens']}}
                write(out/(identifier+'-'+result['arm']+'-calls.json'), client.records)
            else:
                result['actual_new_api_usage'] = {k:0 for k in ['model_calls', 'input_tokens', 'output_tokens']}
            write(out/(identifier+'-'+result['arm']+'-result.json'), result)
            rows.append(result)
            print(json.dumps({'case': identifier, 'arm': result['arm'], 'status': result['status']}), flush=True)

        def export(result, client, report):
            native = present_decision(report)
            result['native_prediction'] = native
            result['checkpoints'] = round_decisions(report)
            if not native['assessment_valid']: raise ValueError('Invalid report')
            answer = json.dumps(native, ensure_ascii=False)
            result['extractor_answer_text'] = answer
            result['extraction'] = extract(client, asdict(target), answer)
            if result['extraction']['decision'] != native['decision']:
                raise ValueError('Extractor changed native decision')
            if any(r.get('error_type') for r in client.records):
                raise ValueError('A model error was swallowed')
            result['prediction'] = native_prediction(report)
            result['status'] = 'completed'

        # Original baseline has its own fresh calls and native source flags.
        result = row('original'); client = make_client('original')
        try:
            agent = NewsTracingAgent(OriginalCorpusClient(client, clean), max_depth=1,
                console=Console(file=io.StringIO(), quiet=True))
            task = DATA_RULE+'\nAssess the claim under this fixed contract:\n'+json.dumps(asdict(target))
            report = asdict(asyncio.run(agent.run(task)))
            write(out/(identifier+'-original-report.json'), report)
            answer = report['direct_response']; result['extractor_answer_text'] = answer
            result['extraction'] = extract(client, asdict(target), answer)
            if any(r.get('error_type') for r in client.records): raise ValueError('Swallowed model error')
            ids = {m.url:m.version_id for m in eligible}
            result['prediction'] = {'decision': result['extraction']['decision'],
                'origins': sorted({ids[s['url']] for s in report['sources'] if s.get('is_original') and s['url'] in ids}),
                'edges': [], 'provenance_evaluable': True, 'origin_evaluable': True,
                'edges_evaluable': False, 'edge_export_limit': 'No native source-to-source edge representation'}
            result['status'] = 'completed'
        except Exception as exc: result['error_type'] = type(exc).__name__
        finally: finish(result, client)

        captures = []; prefix = None; prefix_usage = None; initial_report = None
        result = row('single'); client = make_client('single')
        try:
            provider = ReanalysisProvider(eligible)
            initial_report = p.run_provenance(target, provider, FeedbackDecomposer(client), FeedbackVerifier(client),
                INITIAL, checkpoint_callback=captures.append)
            write(out/(identifier+'-single-report.json'), initial_report)
            write(out/(identifier+'-single-retrieval.json'), provider.history)
            if not captures: raise ValueError('No valid first-round checkpoint')
            cp = captures[-1]
            result['state_sha256'] = p.checkpoint_sha256(cp)
            (out/(identifier+'-shared-checkpoint.json')).write_text(p.canonical_checkpoint_json(cp)+'\n')
            prefix, prefix_usage = deepcopy(client.records), deepcopy(client.usage())
            result['prefix_usage'] = prefix_usage
            export(result, client, initial_report)
        except Exception as exc: result['error_type'] = type(exc).__name__
        finally: finish(result, client)
        if result['status'] != 'completed':
            for arm in ARMS[2:]:
                failed = row(arm); failed['error_type'] = 'InvalidSharedPrefix'; finish(failed, None)
            return rows

        cp = captures[-1]; base_hash = p.checkpoint_sha256(cp)
        branches = ['loop_psi', 'loop_frozen', 'independent']
        random.Random(identifier).shuffle(branches)
        for arm in branches:
            result = row(arm); client = make_client(arm)
            result['shared_state_sha256'] = base_hash
            result['prefix_usage'] = prefix_usage
            try:
                seed_client(client, prefix, prefix_usage)
                if arm in {'loop_psi', 'loop_frozen'}:
                    provider = ReanalysisProvider(eligible)
                    psi = FeedbackDecomposer(client) if arm == 'loop_psi' else FrozenPsi(cp)
                    report = p.run_provenance(target, provider, psi, FeedbackVerifier(client), CONTINUED,
                        checkpoint=cp)
                    write(out/(identifier+'-'+arm+'-report.json'), report)
                    write(out/(identifier+'-'+arm+'-retrieval.json'), provider.history)
                    result['frozen_psi_adapter_returns'] = psi.calls if arm == 'loop_frozen' else 0
                    if p.checkpoint_sha256(cp) != base_hash: raise ValueError('Shared checkpoint mutated')
                    if report['usage']['verification_calls'] != 3: raise ValueError('Three actual verifications required')
                else:
                    trials = [initial_report]
                    for index in [1, 2]:
                        trial = p.run_provenance(target, ReanalysisProvider(eligible), FeedbackDecomposer(client),
                            FeedbackVerifier(client), INITIAL)
                        write(out/(identifier+'-independent-trial-'+str(index)+'.json'), trial)
                        if not trial['assessment_valid']: raise ValueError('Independent trial failed')
                        trials.append(trial)
                    selector_schema = {'type':'object','additionalProperties':False,
                        'properties':{'selected_trial':{'type':'integer','enum':[0,1,2]},'rationale':{'type':'string'}},
                        'required':['selected_trial','rationale']}
                    selection = client.call(DATA_RULE+'\nSelect the best-supported existing trial. Do not create a new answer. '+
                        'Trial order gives no quality information. Explain using supplied evidence.',
                        json.dumps({'target':asdict(target),'materials':[asdict(m) for m in eligible],
                            'trials':[{'prediction':present_decision(t),'verification':t['verification_history'][-1:]} for t in trials]},ensure_ascii=False),
                        selector_schema)
                    if type(selection['selected_trial']) is not int or selection['selected_trial'] not in [0,1,2]:
                        raise ValueError('Invalid selected trial')
                    result['selection'] = selection
                    result['independent_trials'] = [present_decision(t) for t in trials]
                    report = trials[selection['selected_trial']]
                    write(out/(identifier+'-independent-report.json'), report)
                export(result, client, report)
            except Exception as exc: result['error_type'] = type(exc).__name__
            finally: finish(result, client)
        return rows

    cases = deepcopy(data['cases']); random.Random(20260905).shuffle(cases)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one_case, c) for c in cases]):
            results.extend(future.result())
            write(out/'results.json', results)
    status = 'completed' if all(r['status']=='completed' for r in results) else 'has_errors'
    write(out/'status.json', {'status':status,'jobs':len(results),
        'actual_new_api_usage':{k:sum(r['actual_new_api_usage'][k] for r in results)
            for k in ['model_calls','input_tokens','output_tokens']},
        'logical_arm_calls':sum(r['usage']['model_calls'] for r in results)})
    return 0 if status == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','original','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--model',default='gpt-6-astra')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prompt-key',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4: parser.error('workers must be 1..4')
    logging.disable(logging.CRITICAL)
    if args.prompt_key:
        if not sys.stdin.isatty(): raise SystemExit('Echo-disabled terminal required')
        os.environ['OPENAI_API_KEY'] = getpass.getpass('API credential (hidden): ')
    try:return run(args)
    finally:
        if args.prompt_key:os.environ.pop('OPENAI_API_KEY',None)


if __name__ == '__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status':'stopped','error_type':type(exc).__name__}),flush=True)
        raise SystemExit(1)
