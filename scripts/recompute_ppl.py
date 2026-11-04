"""Recompute length-adjusted PPL associations and within-language decile scores.

python scripts/recompute_ppl.py --data data/ppl --output association_results.json
Dependencies: numpy, scipy, statsmodels, threadpoolctl.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.special import expit
import statsmodels.api as sm
from threadpoolctl import threadpool_limits

FEATURES = ('whole', 'block_median', 'block_mean', 'block_max', 'block_min')
LANGUAGES = ('cpp', 'python', 'java')
PROMPTS = ('zs', 'fs', 'cot')


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def fit_logit(x, y, weights, initial=None):
    mean = np.average(y, weights=weights)
    if not 0 < mean < 1:
        return None
    beta = np.zeros(x.shape[1]) if initial is None else initial.copy()
    if initial is None:
        beta[0] = math.log(mean/(1-mean))
    for _ in range(60):
        eta = x @ beta
        mu = expit(eta)
        information = x.T @ ((weights*mu*(1-mu))[:,None]*x)
        gradient = x.T @ (weights*(y-mu))
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(step)) < 1e-8:
            return beta
        old = np.sum(weights*(y*eta-np.logaddexp(0,eta)))
        for _ in range(20):
            candidate = beta+step
            z = x @ candidate
            if np.sum(weights*(y*z-np.logaddexp(0,z))) >= old-1e-10:
                break
            step *= .5
        else:
            return None
        beta = candidate
        if not np.isfinite(beta).all() or np.max(np.abs(beta)) > 40:
            return None
    return None


def recompute(data, repetitions, seed):
    programs = read_rows(data/'program_ppl.jsonl.gz')
    codes = {row['code_key']:row for row in programs}
    pairs = read_rows(data/'pair_ppl_judgments.jsonl.gz')
    assert len(codes)==len(programs)==4455
    assert len({(r['language'],r['pair_id']) for r in pairs})==len(pairs)==2700
    for code in codes.values():
        sizes,ppls = code['block_token_counts'],code['block_ppl']
        assert sum(sizes)==code['tokens']-1 and max(sizes)<=64
        assert len(sizes)==len(ppls)==math.ceil(sum(sizes)/64)
        f=dict(whole=math.exp(math.fsum(n*math.log(p) for n,p in zip(sizes,ppls))/sum(sizes)),
               block_median=statistics.median(ppls),block_mean=statistics.mean(ppls),
               block_max=max(ppls),block_min=min(ppls))
        for key,value in f.items():
            assert math.isclose(value,code['features'][key],rel_tol=1e-12)
    fits,groups = [],[]
    for li,language in enumerate(LANGUAGES):
        rows = [r for r in pairs if r['language']==language]
        assert len(rows)==900
        for row in rows:
            for feature in FEATURES:
                values=[codes[row[side+'_code_key']]['features'][feature] for side in ('slow','fast')]
                assert math.isclose(sum(values)/2,row['features'][feature],rel_tol=1e-12)
        clusters=np.array([r['problem_id'] for r in rows])
        labels,inverse=np.unique(clusters,return_inverse=True)
        draws=np.random.default_rng(seed+li).multinomial(len(labels),np.ones(len(labels))/len(labels),size=repetitions)
        weights=draws[:,inverse].astype(float)
        controls=np.array([[math.log((r['slow_tokens']+r['fast_tokens'])/2),
                           abs(math.log(r['slow_tokens']/r['fast_tokens']))] for r in rows])
        assert controls.shape==(900,2)
        for feature in FEATURES:
            values=np.array([r['features'][feature] for r in rows])
            columns=np.column_stack((np.log(values),controls))
            x=np.column_stack((np.ones(900),(columns-columns.mean(axis=0))/columns.std(axis=0)))
            assert x.shape==(900,4) and np.linalg.matrix_rank(x)==4
            edges=np.unique(np.quantile(values,np.arange(1,10)/10))
            edges=edges[(edges>values.min()) & (edges<values.max())]
            bins=np.searchsorted(edges,values,side='left')
            for prompt in PROMPTS:
                orders=np.array([r['outcomes'][prompt] for r in rows])
                assert orders.shape==(900,2) and set(orders.ravel())<={0,1}
                outcomes={'accuracy':orders.mean(axis=1),'bc':orders.prod(axis=1)}
                for target,y in outcomes.items():
                    beta=fit_logit(x,y,np.ones(900))
                    assert beta is not None
                    reference=sm.GLM(y,x,family=sm.families.Binomial()).fit()
                    assert np.max(np.abs(reference.params-beta))<1e-7
                    boot=[]
                    for w in weights:
                        estimate=fit_logit(x,y,w,beta)
                        if estimate is not None:
                            boot.append(math.exp(estimate[1]))
                    assert len(boot)==repetitions
                    fits.append(dict(language=language,feature=feature,prompt=prompt,target=target,
                      coefficient_per_sd=float(beta[1]),odds_ratio=math.exp(beta[1]),
                      ci95=np.quantile(boot,[.025,.975]).tolist(),
                      bootstrap_requested=repetitions,bootstrap_valid=len(boot)))
                    for b in range(bins.max()+1):
                        mask=bins==b
                        denom=weights[:,mask].sum(axis=1)
                        means=(weights[:,mask] @ y[mask])[denom>0]/denom[denom>0]
                        groups.append(dict(language=language,feature=feature,prompt=prompt,target=target,
                          decile=b+1,pairs=int(mask.sum()),mean=float(y[mask].mean()),
                          ci95=np.quantile(means,[.025,.975]).tolist()))
                print(language,feature,prompt,flush=True)
    return dict(protocol=dict(model='Qwen3.8-27B',programs=4455,pairs=2700,
      window=1024,stride=256,block_max_tokens=64,
      block_layout='contiguous balanced blocks of scored tokens',
      pair_aggregation='arithmetic mean of two program-level feature values',
      features=list(FEATURES),controls=['log mean token length','absolute log token length ratio'],
      regression='fractional-binomial logit for pair Accuracy; binary logistic for BC',
      predictor='within-language standardized log pair PPL',odds_ratio='exp(coefficient of standardized log pair PPL)',
      bootstrap_repetitions=repetitions,bootstrap_unit='problem within language, retaining every pair and both orders',
      seeds={l:seed+i for i,l in enumerate(LANGUAGES)},
      fixed=['decile membership','predictor centering and scaling'],
      ci='pointwise percentile 95%; 2.5th and 97.5th percentiles'),
      regressions=fits,deciles=groups)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/ppl')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    assert args.repetitions>0 and not args.output.exists()
    with threadpool_limits(limits=1):
        result=recompute(args.data,args.repetitions,args.seed)
    with args.output.open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
