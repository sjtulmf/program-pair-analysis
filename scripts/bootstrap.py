import collections,json
import numpy as np
from data_tools import ROOT,rows,pairs
def calculate():
    data=pairs();groups=collections.defaultdict(dict)
    for r in rows(ROOT/'data/main_judgments.jsonl.gz'):
        k=r['model'],r['mode'];key=r['language'],r['pair_id']
        groups[k].setdefault(key,{})[r['direction']]=r['correct']
    baseline=collections.defaultdict(dict)
    for r in rows(ROOT/'data/baseline_predictions.jsonl.gz'):
        baseline[r['rule']][r['language'],r['pair_id']]=(r['forward_correct'],r['reverse_correct'])
    names=sorted(groups);bnames=sorted(baseline);point=[];draw=[];by={};rng=np.random.default_rng(42)
    def cell(v,d):
        ci=np.quantile(d,[.025,.975],axis=0,method='linear')
        return dict(accuracy=float(v[0]),bc=float(v[1]),accuracy_ci95=ci[:,0].tolist(),bc_ci95=ci[:,1].tolist())
    for l in ('cpp','python','java'):
        ps=sorted(p for lang,p in data if lang==l);tasks=sorted({data[l,p]['problem_id'] for p in ps});ix={p:i for i,p in enumerate(tasks)}
        labels=[ix[data[l,p]['problem_id']] for p in ps]
        ys=[]
        for p in ps:
            row=[]
            for k in names:
                a,b=(groups[k][l,p][o] for o in ('forward','reverse'));row.append(((int(a)+int(b))/2,int(a and b)))
            for k in bnames:
                a,b=baseline[k][l,p];row.append(((int(a)+int(b))/2,int(a and b)))
            ys.append(row)
        y=np.array(ys);sums=np.zeros((len(tasks),len(names)+len(bnames),2));np.add.at(sums,labels,y)
        w=rng.multinomial(len(tasks),np.full(len(tasks),1/len(tasks)),size=2000)
        d=((w@sums.reshape(len(tasks),-1))/(w@np.bincount(labels))[:,None]).reshape(2000,len(names)+len(bnames),2)
        p=y.mean(axis=0);point.append(p);draw.append(d);by[l]=[cell(v,d[:,i]) for i,v in enumerate(p)]
    p,d=np.mean(point,axis=0),np.mean(draw,axis=0);out=[]
    for i,k in enumerate(names+[(b,'') for b in bnames]):
        out.append(dict(model_or_rule=k[0],mode=k[1],**cell(p[i],d[:,i]),languages={l:by[l][i] for l in by}))
    for mode in ('zs','fs','cot'):
        ix=[i for i,k in enumerate(names) if k[1]==mode]
        out.append(dict(model_or_rule='average',mode=mode,**cell(p[ix].mean(axis=0),d[:,ix].mean(axis=1))))
    estimates = {}
    for mode in ('zs', 'fs', 'cot'):
        indices = [i for i, key in enumerate(names) if key[1] == mode]
        pv, dv = p[indices].mean(axis=0), d[:, indices].mean(axis=1)
        estimates[mode] = (pv, dv)
        gap = dv[:, 0] - dv[:, 1]
        out.append(dict(model_or_rule='accuracy_minus_bc', mode=mode,
                        gap=float(pv[0]-pv[1]), ci95=np.quantile(gap,[.025,.975]).tolist()))
    for mode in ('fs', 'cot'):
        pv, dv = estimates[mode]
        ref, ref_draw = estimates['zs']
        delta = dv - ref_draw
        gap = delta[:, 0] - delta[:, 1]
        out.append(dict(model_or_rule='paired_difference_from_zs', mode=mode,
                        **cell(pv-ref,delta), gap=float((pv-ref)[0]-(pv-ref)[1]),
                        gap_ci95=np.quantile(gap,[.025,.975]).tolist()))
    return out
if __name__=='__main__':print(json.dumps(calculate(),indent=2))
