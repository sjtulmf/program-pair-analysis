import argparse,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def rows(path):
    with gzip.open(path,'rt',encoding='utf8') as f:yield from map(json.loads,f)
def pairs():
    return {(x['language'],x['pair_id']):x for x in rows(ROOT/'data/pairs.jsonl.gz')}
def dataset():
    codes={(r['language'],r['code_id']):r['code'] for r in rows(ROOT/'data/code.jsonl.gz')}
    for (l,p),x in pairs().items():
        z=dict(x)
        for side in ('slow_code','fast_code','measured_slow_code','measured_fast_code'):
            z[side]=codes[l,x[side+'_id']]
        yield z
def make_prompt(language,pair_id,mode,direction):
    import copy,re
    spec=json.loads((ROOT/'data/prompts.json').read_bytes());p=next(x for x in dataset() if x['language']==language and x['pair_id']==pair_id)
    messages=copy.deepcopy(spec['skeletons'][language+'/'+mode])
    assert direction in ('forward','reverse')
    order=['slow_code','fast_code'] if direction=='forward' else ['fast_code','slow_code']
    replacement=dict(zip(spec['markers'],(p[k] for k in order)))
    messages[1]['content']=re.sub('|'.join(map(re.escape,spec['markers'])),lambda m:replacement[m[0]],messages[1]['content'])
    return messages
def baseline():
    f={(r['language'],r['code_id']):r for r in rows(ROOT/'data/baseline_features.jsonl.gz')}
    rules={'fewer_lines':('raw_nloc',),'fewer_ast_nodes':('ast_nodes',),'shallower_loops':('num_nested_loop',),
           'fewer_io_sites':('io_sites',),'combined_vote':('raw_nloc','analysis_ccn_max','num_nested_loop','num_loop','num_if')}
    for (l,p),r in pairs().items():
        a,b=f[l,r['slow_code_id']],f[l,r['fast_code_id']]
        for name,ks in rules.items():
            av=[a[k] for k in ks];bv=[b[k] for k in ks];available=all(v is not None for v in av+bv)
            vote=sum((x<y)-(x>y) for x,y in zip(av,bv)) if available else 0
            if vote:pred='A' if vote>0 else 'B'
            else:
                key=lambda v:hashlib.sha256(('42:'+v['code_id']).encode()).digest()
                pred='A' if key(a)<=key(b) else 'B'
            yield dict(rule=name,language=l,pair_id=p,problem_id=r['problem_id'],available=available,tie=vote==0,
                       forward_prediction=pred,reverse_prediction='B' if pred=='A' else 'A',
                       forward_correct=pred=='B',reverse_correct=pred=='B',both_correct=pred=='B')
def verify():
    spec=json.loads((ROOT/'index.json').read_bytes())
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not any(part.startswith('.') for part in p.relative_to(ROOT).parts)}
    assert actual==set(spec['files'])|{'index.json'}
    cache={}
    def keyset(n,fields):
        k=n,tuple(fields)
        if k not in cache:cache[k]={tuple(r[f] for f in fields) for r in rows(ROOT/n)}
        return cache[k]
    for n,meta in spec['files'].items():
        with (ROOT/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['sha256']
        if 'rows' in meta:
            key=meta['primary_key'];seen=set();count=0
            for r in rows(ROOT/n):
                k=tuple(r[f] for f in key);assert k not in seen,(n,k);seen.add(k);count+=1
            assert count==meta['rows']
    for n,fields,target,keys in spec['links']:assert keyset(n,fields)<=keyset(target,keys)
    expected={(r['rule'],r['language'],r['pair_id']):r for r in rows(ROOT/'data/baseline_predictions.jsonl.gz')}
    for r in baseline():assert expected.pop((r['rule'],r['language'],r['pair_id']))==r
    assert not expected
    data=pairs();conditions=set();judgments={};models=set()
    for r in rows(ROOT/'data/main_judgments.jsonl.gz'):
        k=(r['model'],r['language'],r['pair_id'],r['mode'],r['direction'])
        assert k not in conditions;conditions.add(k);models.add(r['model']);judgments[r['result_id']]=r
        assert r['problem_id']==data[r['language'],r['pair_id']]['problem_id']
        assert r['expected_answer']==('B' if r['direction']=='forward' else 'A')
        assert r['correct']==(r['parse_status']=='ok' and r['prediction']==r['expected_answer'])
    assert len(models)==6 and conditions=={(m,l,p,s,d) for m in models for l,p in data for s in ('zs','fs','cot') for d in ('forward','reverse')}
    assert {(r['language'],r['pair_id']) for r in rows(ROOT/'data/complexity.jsonl.gz')}==set(data)
    for r in rows(ROOT/'data/error_analysis.jsonl.gz'):
        j=judgments[r['result_id']]
        assert j['mode']=='cot' and not j['correct'] and j['parse_status']=='ok' and j['finish_reason']=='stop'
        assert all(r[k]==j[k] for k in ('language','pair_id','problem_id','direction'))
    variants={};slots=set()
    for r in rows(ROOT/'data/perturbation_variants.jsonl.gz'):
        k=(r['language'],r['pair_id'],r['operator'],r['site'])
        assert k not in slots;slots.add(k);variants[r['language'],r['variant_id']]=r
    answers=set()
    for r in rows(ROOT/'data/perturbation_judgments.jsonl.gz'):
        k=(r['language'],r['variant_id'],r['direction']);assert k not in answers;answers.add(k)
        v=variants[r['language'],r['variant_id']]
        assert all(r[k]==v[k] for k in ('pair_id','operator','site'))
    assert answers=={(l,v,d) for l,v in variants for d in ('forward','reverse')}
    for r in rows(ROOT/'data/code.jsonl.gz'):
        assert r['code'] is None or hashlib.sha256(r['code'].encode()).hexdigest()==r['code_id']
    return spec['counts']
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('dataset','baseline','verify','prompt'))
    p.add_argument('--language');p.add_argument('--pair');p.add_argument('--mode',choices=('zs','fs','cot'));p.add_argument('--direction',choices=('forward','reverse'))
    a=p.parse_args()
    if a.action in ('verify','prompt'):
        print(json.dumps(verify() if a.action=='verify' else make_prompt(a.language,a.pair,a.mode,a.direction),ensure_ascii=True))
    else:
        for r in {'dataset':dataset,'baseline':baseline}[a.action]():print(json.dumps(r,ensure_ascii=True))
