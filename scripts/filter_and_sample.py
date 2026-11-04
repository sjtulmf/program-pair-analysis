import argparse,collections,gzip,hashlib,heapq,json,math,re
SEED=42
STRATA=('2-4','4-8','8+')
def seed_rank(value):
    return hashlib.sha256((str(SEED) + ':' + value).encode()).hexdigest()


def normalize(code):
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'[ \t]+', ' ', code)
    return re.sub(r'\n\s*\n', '\n', code).strip()


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def stratum(speedup):
    if speedup < 2 or not math.isfinite(speedup):
        raise ValueError('Ineligible speedup')
    return '2-4' if speedup < 4 else '4-8' if speedup < 8 else '8+'


class Flow:
    """Integral successive-shortest-path min-cost max-flow, deterministic."""
    def __init__(self, n):
        self.g = [[] for _ in range(n)]

    def add(self, u, v, cap, cost):
        edge = [v, len(self.g[v]), cap, cost, cap]
        self.g[u].append(edge)
        self.g[v].append([u, len(self.g[u])-1, 0, -cost, 0])
        return edge

    def run(self, source, sink, target):
        n = len(self.g)
        dist = [None] * n
        dist[source] = 0
        queue = collections.deque([source])
        queued = {source}
        while queue:
            u = queue.popleft(); queued.remove(u)
            for v, _, cap, cost, _ in self.g[u]:
                if cap and (dist[v] is None or dist[v] > dist[u] + cost):
                    dist[v] = dist[u] + cost
                    if v not in queued:
                        queued.add(v); queue.append(v)
        potential = [d or 0 for d in dist]
        total, objective = 0, 0
        while total < target:
            dist = [None] * n; dist[source] = 0
            parent = [None] * n; heap = [(0, source)]
            while heap:
                d, u = heapq.heappop(heap)
                if d != dist[u]:
                    continue
                for i, e in enumerate(self.g[u]):
                    v, _, cap, cost, _ = e
                    if not cap:
                        continue
                    reduced = cost + potential[u] - potential[v]
                    assert reduced >= 0
                    nd = d + reduced
                    if dist[v] is None or nd < dist[v]:
                        dist[v] = nd; parent[v] = (u, i)
                        heapq.heappush(heap, (nd, v))
            if parent[sink] is None:
                break
            for i, d in enumerate(dist):
                if d is not None:
                    potential[i] += d
            v = sink
            while v != source:
                u, i = parent[v]; e = self.g[u][i]
                e[2] -= 1; self.g[v][e[1]][2] += 1
                objective += e[3]; v = u
            total += 1
        return total, objective


def select(pool, cap, fine_bins=10, target_per_stratum=300):
    # Within each observed stratum range, use equal-width log2 bins.
    ranges = {s: (min(math.log2(p['speedup']) for p in pool if p['stratum'] == s),
                  max(math.log2(p['speedup']) for p in pool if p['stratum'] == s)) for s in STRATA}
    buckets = collections.defaultdict(list)
    for p in pool:
        lo, hi = ranges[p['stratum']]
        b = min(fine_bins-1, int((math.log2(p['speedup'])-lo)/(hi-lo)*fine_bins)) if hi > lo else 0
        buckets[p['problem_id'], p['stratum'], b].append(p)
    pids = sorted({p['problem_id'] for p in pool}, key=seed_rank)
    labels = [('issue', p) for p in pids]
    labels += [('bin', s, b) for s in STRATA for b in range(fine_bins)]
    labels += [('stratum', s) for s in STRATA]
    ids = {v: i+2 for i, v in enumerate(labels)}
    flow = Flow(len(ids)+2)
    source, sink = 0, 1
    # Lexicographic: maximum coverage, then minimum squared bin counts,
    # then minimum squared per-problem counts. Fixed seeds settle ties.
    for pid in pids:
        for k in range(1, cap+1):
            flow.add(source, ids['issue', pid], 1, 2*k-1-(10**16 if k == 1 else 0))
    refs = {}
    for key in sorted(buckets, key=lambda k: (seed_rank(k[0]), k[1], k[2])):
        pid, s, b = key
        refs[key] = flow.add(ids['issue', pid], ids['bin', s, b], len(buckets[key]), 0)
    for s in STRATA:
        for b in range(fine_bins):
            for k in range(1, target_per_stratum+1):
                flow.add(ids['bin', s, b], ids['stratum', s], 1, (2*k-1)*10**6)
        flow.add(ids['stratum', s], sink, target_per_stratum, 0)
    count, cost = flow.run(source, sink, target_per_stratum*len(STRATA))
    selected = []
    for key, edge in refs.items():
        take = edge[4]-edge[2]
        selected.extend(dict(p, fine_bin=key[2]) for p in sorted(buckets[key], key=lambda p: p.get('sampling_key') or seed_rank(p['pair_id']))[:take])
    assert len(selected) == count
    selected.sort(key=lambda p: p.get('sampling_key') or seed_rank(p['pair_id']))
    return selected, {'per_problem_cap': cap, 'fine_bins': fine_bins,
                      'log2_ranges': ranges, 'objective_integer': cost,
                      'available_per_bin': {s: [sum(len(v) for (pid, ss, bb), v in buckets.items() if ss == s and bb == b)
                                               for b in range(fine_bins)] for s in STRATA}}



def construct(candidates,cap=5):
    kept=[];seen=set();grouped=collections.defaultdict(list);tokens={};norms={}
    for p in candidates:
        assert p['pair_id'] not in seen;seen.add(p['pair_id'])
        for side in ('slow','fast'):
            h=hashlib.sha256(p[side+'_code'].encode()).hexdigest()
            norms[h]=normalize(p[side+'_code']);tokens[h]=frozenset(re.findall(r'\b\w+\b',norms[h]))
            p[side+'_hash']=h
        p['stratum']=stratum(p['speedup'])
    def similar(a,b):return norms[a]==norms[b] or jaccard(tokens[a],tokens[b])>=0.9
    for p in sorted(candidates,key=lambda p:p.get('sampling_key') or seed_rank(p['pair_id'])):
        if jaccard(tokens[p['slow_hash']],tokens[p['fast_hash']])>=0.9:continue
        g=grouped[p['problem_id'],p['stratum']]
        if any(similar(p['slow_hash'],q['slow_hash']) and similar(p['fast_hash'],q['fast_hash']) for q in g):continue
        kept.append(p);g.append(p)
    selected,_=select(kept,cap=cap)
    assert len(selected)==900 and collections.Counter(p['stratum'] for p in selected)=={s:300 for s in STRATA}
    for p in selected:
        del p['slow_hash'];del p['fast_hash']
    return selected
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('candidates');parser.add_argument('--cap',type=int,default=5)
    args=parser.parse_args()
    opener=gzip.open if args.candidates.endswith('.gz') else open
    with opener(args.candidates,'rt',encoding='utf8') as f:data=[json.loads(l) for l in f if l.strip()]
    for r in construct(data,args.cap):print(json.dumps(r,ensure_ascii=True))
