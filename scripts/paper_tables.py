"""Recompute paper point estimates from canonical records. Run from any directory."""
import argparse
from collections import Counter, defaultdict
import json
from statistics import mean
from data_tools import ROOT, rows


def load(name):
    return rows(ROOT / name)


def pct(values):
    return 100 * mean(values)


def scores(pairs):
    pairs = list(pairs)
    return {'accuracy': pct(mean(p) for p in pairs), 'bc': pct(all(p) for p in pairs)}


def main():
    records = list(load('data/pairs.jsonl.gz'))
    pairs = {(r['language'], r['pair_id']): r for r in records}
    assert len(pairs) == len(records) == 2700
    judgments = {}
    by_model, by_language_prompt, by_bin, by_prompt = (defaultdict(list) for _ in range(4))
    for r in load('data/main_judgments.jsonl.gz'):
        key = (r['language'], r['pair_id'], r['model'], r['mode'])
        slot = judgments.setdefault(key, {})
        assert r['direction'] not in slot and key[:2] in pairs
        slot[r['direction']] = bool(r['correct'])
    assert len(judgments) == 48600
    complexity = {(r['language'], r['pair_id']): r['complexity_group'] for r in load('data/complexity.jsonl.gz')}
    model_language_prompt = defaultdict(list)
    by_complexity = defaultdict(list)
    by_model_complexity, by_model_prompt = defaultdict(list), defaultdict(list)
    for (lang, pair, model, mode), slot in judgments.items():
        assert set(slot) == {'forward', 'reverse'}
        value = tuple(slot[d] for d in ('forward', 'reverse'))
        model_language_prompt[model, lang, mode].append(value)
        by_model[model].append(value)
        by_language_prompt[lang, mode].append(value)
        by_bin[lang, pairs[lang, pair]['stratum']].append(value)
        by_prompt[mode].append(value)
        by_complexity[lang, complexity[lang, pair]].append(value)
        by_model_complexity[lang, model, complexity[lang, pair]].append(value)
        by_model_prompt[model, mode].append(value)
    repeat = {}
    for r in load('data/repeat_judgments.jsonl.gz'):
        key = r['language'], r['pair_id'], r['direction']
        assert key not in repeat and r['model'] == 'glm-5.3-flash' and r['mode'] == 'zs'
        repeat[key] = r['correct']
    assert len(repeat) == 4790
    weights = Counter((r['language'], r['pair_id']) for r in load('index/perturbation_cases.jsonl.gz'))
    assert {k[:2] for k in repeat} == set(weights) and len(weights) == 2395
    control = {}
    for lang in sorted({k[0] for k in weights}):
        old, new, retained = [], [], []
        for (l, p), w in weights.items():
            if l != lang:
                continue
            original = judgments[l, p, 'glm-5.3-flash', 'zs']
            a = tuple(original[d] for d in ('forward', 'reverse'))
            b = tuple(repeat[l, p, d] for d in ('forward', 'reverse'))
            old.extend([a] * w)
            new.extend([b] * w)
            if all(a):
                retained.extend([all(b)] * w)
        control[lang] = {'matched_cases': len(old), 'original': scores(old),
                         'repeat': scores(new), 'cc_retained': pct(retained)}
    groups = Counter((r['language'], r['complexity_group']) for r in load('data/complexity.jsonl.gz'))
    errors = list(load('data/error_analysis.jsonl.gz'))
    categories = Counter(r['classification']['category'] for r in errors)
    assert sum(categories.values()) == 5370
    data = {}
    for lang in ('cpp', 'python', 'java'):
        subset = [r for r in records if r['language'] == lang]
        data[lang] = {'pairs': len(subset), 'problems': len({r['problem_id'] for r in subset}),
                      'measured_implementations': len({(r['problem_id'], r[k+'_id']) for r in subset for k in ('slow', 'fast')}),
                      'displayed_sources': len({r[k+'_code_id'] for r in subset for k in ('slow', 'fast')})}
    figure5 = json.loads((ROOT / 'data/figure5_case.json').read_text(encoding='utf8'))
    def aggregate(group):
        return {'/'.join(k) if isinstance(k, tuple) else k: scores(v) for k, v in sorted(group.items())}
    return {'dataset': data, 'model_means': aggregate(by_model),
            'overall': scores(x for v in by_model.values() for x in v),
            'model_language_prompt': aggregate(model_language_prompt),
            'model_by_prompt': aggregate(by_model_prompt), 'complexity_by_model': aggregate(by_model_complexity),
            'language_by_prompt': aggregate(by_language_prompt), 'language_by_ratio': aggregate(by_bin),
            'prompt_means': aggregate(by_prompt), 'unedited_repeat': control,
            'complexity': {'/'.join(k): dict(scores(v), pairs=groups[k]) for k, v in sorted(by_complexity.items())},
            'error_categories': dict(sorted(categories.items())),
            'figure5_pair_id': figure5['pair_id']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(main(), indent=2, sort_keys=True))
