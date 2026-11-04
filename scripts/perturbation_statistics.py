"""Summarize frozen GLM perturbation judgments without running any models.

Operator comparisons use identical (base pair, edited side) cases within
each language. All scores refer to the ORIGINAL measured efficiency label;
this script does not certify or change labels for transformed programs.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


LANGUAGES = ('cpp', 'python', 'java')
OPERATORS = ('IOE', 'IS', 'OOS', 'DBI', 'HVI')
SIDES = ('slow_only', 'fast_only', 'both')
GOLD = {'forward': 'B', 'reverse': 'A'}
ORDER_STATES = ('CC', 'CW', 'WC', 'WW')


def records(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def aggregate(rows, baseline):
    assert rows
    before = after = before_bc = after_bc = unchanged = 0
    transitions, bc_transitions, correct_orders = Counter(), Counter(), Counter()
    order_states = Counter()
    state_pairs = defaultdict(set)
    for row in rows:
        before_flags, after_flags = [], []
        for order, gold in GOLD.items():
            old = baseline[row['language'], row['pair_id'], order]
            new = row['predictions'][order]
            old_ok, new_ok = old == gold, new == gold
            before_flags.append(old_ok)
            after_flags.append(new_ok)
            before += old_ok
            after += new_ok
            unchanged += old == new
            transitions[f'{int(old_ok)}{int(new_ok)}'] += 1
        old_bc, new_bc = all(before_flags), all(after_flags)
        old_state = ''.join('C' if flag else 'W' for flag in before_flags)
        new_state = ''.join('C' if flag else 'W' for flag in after_flags)
        order_states[old_state, new_state] += 1
        state_pairs[old_state].add((row['language'], row['pair_id']))
        before_bc += old_bc
        after_bc += new_bc
        bc_transitions[f'{int(old_bc)}{int(new_bc)}'] += 1
        correct_orders[sum(before_flags), sum(after_flags)] += 1
    variants = len(rows)
    ordered = 2 * variants
    assert sum(transitions.values()) == ordered
    assert unchanged == transitions['00'] + transitions['11']
    assert after - before == transitions['01'] - transitions['10']
    lost, recovered = bc_transitions['10'], bc_transitions['01']
    assert after_bc - before_bc == recovered - lost
    assert lost == correct_orders[2, 0] + correct_orders[2, 1]
    assert sum(order_states.values()) == variants
    assert before == sum(b.count('C') * n for (b, a), n in order_states.items())
    assert after == sum(a.count('C') * n for (b, a), n in order_states.items())
    unchanged_states = sum(order_states[s, s] for s in ORDER_STATES)
    both_changed = sum(n for (b, a), n in order_states.items()
                       if sum(x != y for x, y in zip(b, a)) == 2)
    one_changed = variants - unchanged_states - both_changed
    assert unchanged == 2 * unchanged_states + one_changed
    return dict(
        variants=variants, ordered_judgments=ordered,
        base_pairs=len({(r['language'], r['pair_id']) for r in rows}),
        retained_choices=unchanged, retention_percent=100 * unchanged / ordered,
        original_agreements=before, transformed_agreements=after,
        original_agreement_percent=100 * before / ordered,
        transformed_agreement_percent=100 * after / ordered,
        agreement_change_pp=100 * (after - before) / ordered,
        original_bc_count=before_bc, transformed_bc_count=after_bc,
        original_bc_percent=100 * before_bc / variants,
        transformed_bc_percent=100 * after_bc / variants,
        bc_change_pp=100 * (after_bc - before_bc) / variants,
        bc_lost=lost, bc_recovered=recovered,
        bc_loss_percent=100 * lost / before_bc if before_bc else None,
        bc_recovery_percent=100 * recovered / (variants - before_bc)
        if variants > before_bc else None,
        bc_transitions={name: bc_transitions[code] for name, code in (
            ('incorrect_to_incorrect', '00'), ('incorrect_to_correct', '01'),
            ('correct_to_incorrect', '10'), ('correct_to_correct', '11'))},
        correct_order_transitions={str(b): {str(a): correct_orders[b, a]
                                          for a in range(3)} for b in range(3)},
        order_state_transitions={b: {a: order_states[b, a] for a in ORDER_STATES}
                                 for b in ORDER_STATES},
        original_order_state_support={b: dict(
            variants=sum(order_states[b, a] for a in ORDER_STATES),
            base_pairs=len(state_pairs[b])) for b in ORDER_STATES},
        partial_order_reversals=order_states['CW', 'WC'] + order_states['WC', 'CW'],
        predictions_changed_per_variant=dict(
            neither=unchanged_states, one=one_changed, both=both_changed),
        transitions={name: transitions[code] for name, code in (
            ('incorrect_to_incorrect', '00'), ('incorrect_to_correct', '01'),
            ('correct_to_incorrect', '10'), ('correct_to_correct', '11'))})


def pair_bc_loss(rows, baseline):
    """One vote per base pair, with exactly the same 15 variants available."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['language'], row['pair_id']].append(row)
    required = {(op, side) for op in OPERATORS for side in SIDES}
    eligible, originally_bc, lost = [], [], []
    for key, group in sorted(groups.items()):
        if {(r['operator'], r['site']) for r in group} != required:
            continue
        assert len(group) == 15
        eligible.append(key)
        if not all(baseline[key[0], key[1], order] == gold for order, gold in GOLD.items()):
            continue
        originally_bc.append(key)
        if any(not all(r['predictions'][order] == gold for order, gold in GOLD.items())
               for r in group):
            lost.append(key)
    return dict(eligible_pairs=len(eligible), original_bc_pairs=len(originally_bc),
                lost_bc_pairs=len(lost), retained_bc_pairs=len(originally_bc) - len(lost),
                loss_percent=100 * len(lost) / len(originally_bc) if originally_bc else None,
                eligible_membership=eligible, original_bc_membership=originally_bc,
                lost_bc_membership=lost)


def calculate(data_root):
    files = ('data/main_judgments.jsonl.gz', 'data/perturbation_judgments.jsonl.gz',
             'data/perturbation_applicability.jsonl.gz')
    pins = {name: digest(data_root / name) for name in files}
    baseline = {}
    for row in records(data_root / files[0]):
        if (row['model'], row['mode']) != ('glm-5.3-flash', 'zs'):
            continue
        key = row['language'], row['pair_id'], row['direction']
        assert key not in baseline
        assert row['parse_status'] == 'ok' and row['prediction'] in ('A', 'B')
        assert row['expected_answer'] == GOLD[row['direction']]
        assert row['correct'] == (row['prediction'] == row['expected_answer'])
        baseline[key] = row['prediction']
    assert len(baseline) == 5400

    variants = {}
    for row in records(data_root / files[1]):
        assert row['model'] == 'glm-5.3-flash' and row['mode'] == 'zs'
        assert row['parse_status'] == 'ok' and row['prediction'] in ('A', 'B')
        assert row['operator'] in OPERATORS and row['site'] in SIDES
        key = row['language'], row['variant_id']
        value = variants.setdefault(key, dict(
            language=row['language'], pair_id=row['pair_id'],
            operator=row['operator'], site=row['site'], predictions={}))
        assert (value['pair_id'], value['operator'], value['site']) == (
            row['pair_id'], row['operator'], row['site'])
        assert row['direction'] not in value['predictions']
        value['predictions'][row['direction']] = row['prediction']
    assert len(variants) == 35783

    logical = {}
    for key, row in variants.items():
        assert set(row['predictions']) == set(GOLD)
        case = row['language'], row['pair_id'], row['operator'], row['site']
        assert case not in logical
        logical[case] = key
        assert all((row['language'], row['pair_id'], o) in baseline for o in GOLD)
    available = {}
    applicability = Counter()
    pair_slots = defaultdict(Counter)
    for row in records(data_root / files[2]):
        applicability[row['language'], row['operator'], row['status']] += 1
        pair_slots[row['language'], row['pair_id']][row['status']] += 1
        if row['status'] == 'available':
            case = row['language'], row['pair_id'], row['operator'], row['site']
            assert case not in available
            available[case] = row['language'], row['variant_id']
    assert logical == available

    values = list(variants.values())
    full = aggregate(values, baseline)
    assert full['base_pairs'] == 2684 and full['retained_choices'] == 65805
    assert full['transitions'] == dict(
        incorrect_to_incorrect=10964, incorrect_to_correct=2902,
        correct_to_incorrect=2859, correct_to_correct=54841)
    by_language, by_operator, matched, common_cases = {}, {}, {}, {}
    pair_results = {}
    expected = {
        'cpp': (11974, 884, '77.53', '78.57', '70.85', '71.77', 1909, 807),
        'python': (10767, 900, '85.74', '85.95', '81.18', '80.64', 1208, 696),
        'java': (13042, 900, '79.24', '78.28', '72.60', '70.73', 2498, 892)}
    for language in LANGUAGES:
        group = [r for r in values if r['language'] == language]
        total = by_language[language] = aggregate(group, baseline)
        e = expected[language]
        assert (total['variants'], total['base_pairs']) == e[:2]
        assert tuple(f"{total[k]:.2f}" for k in (
            'original_agreement_percent', 'transformed_agreement_percent',
            'original_bc_percent', 'transformed_bc_percent')) == e[2:6]
        groups = {op: [r for r in group if r['operator'] == op] for op in OPERATORS}
        supports = [{(r['pair_id'], r['site']) for r in groups[op]} for op in OPERATORS]
        common = set.intersection(*supports)
        assert (len(common), len({pair for pair, side in common})) == e[6:]
        common_cases[language] = dict(
            cases=len(common), base_pairs=len({pair for pair, side in common}),
            membership=sorted(common))
        by_operator[language], matched[language] = {}, {}
        for operator in OPERATORS:
            by_operator[language][operator] = aggregate(groups[operator], baseline)
            selected = [r for r in groups[operator] if (r['pair_id'], r['site']) in common]
            matched[language][operator] = aggregate(selected, baseline)
            assert matched[language][operator]['variants'] == len(common)
        assert len({matched[language][op]['original_agreements'] for op in OPERATORS}) == 1
        assert len({matched[language][op]['original_bc_count'] for op in OPERATORS}) == 1
        pair_results[language] = pair_bc_loss(group, baseline)
    pair_overall = pair_bc_loss(values, baseline)
    assert (pair_overall['eligible_pairs'], pair_overall['original_bc_pairs'],
            pair_overall['lost_bc_pairs']) == (1610, 1186, 298)
    for language, expected_counts in (
            ('cpp', (551, 382, 109)), ('python', (256, 215, 44)),
            ('java', (803, 589, 145))):
        assert tuple(pair_results[language][k] for k in (
            'eligible_pairs', 'original_bc_pairs', 'lost_bc_pairs')) == expected_counts
    assert (full['original_bc_count'], full['transformed_bc_count'],
            full['bc_lost'], full['bc_recovered']) == (26693, 26502, 1713, 1522)
    assert pins == {name: digest(data_root / name) for name in files}
    coverage = {}
    for language in LANGUAGES:
        slots = {p: v for (lang, p), v in pair_slots.items() if lang == language}
        assert len(slots) == 900 and all(sum(v.values()) == 15 for v in slots.values())
        missing = {p: v for p, v in slots.items() if not v['available']}
        coverage[language] = dict(
            benchmark_pairs=900, included_pairs=900-len(missing),
            pairs_without_variants={p: dict(sorted(v.items())) for p, v in sorted(missing.items())},
            by_operator={op: {status: count for (lang, oper, status), count in sorted(applicability.items())
                              if lang == language and oper == op} for op in OPERATORS})
    return dict(source_sha256=pins, full=full, by_language=by_language,
                by_language_operator=by_operator, common_cases=common_cases,
                matched_by_language_operator=matched,
                pair_bc_loss=dict(overall=pair_overall, by_language=pair_results),
                coverage=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = calculate(args.data_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print('Verified full pool:', json.dumps(report['full']))
    for operator in OPERATORS:
        print(operator, ' | '.join(
            f"{language}: {s['retention_percent']:.2f}, {s['agreement_change_pp']:+.2f}, {s['bc_change_pp']:+.2f}"
            for language in LANGUAGES
            for s in [report['matched_by_language_operator'][language][operator]]))
        print('BC rates', operator, ' | '.join(
            f"{language}: loss={s['bc_loss_percent']:.2f}, recovery={s['bc_recovery_percent']:.2f}"
            for language in LANGUAGES
            for s in [report['matched_by_language_operator'][language][operator]]))
    print('Unique pairs:', json.dumps({k: v for k, v in report['pair_bc_loss']['overall'].items()
                                      if not k.endswith('_membership')}))


if __name__ == '__main__':
    main()
