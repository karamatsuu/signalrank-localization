"""Step 2: LLM-free, download-free file localization on SWE-bench Lite.
Every ranker sees ONLY the issue text and the repo's path list (+ byte sizes).
Outputs results/*.json and figures used in the paper."""
import json, os, re, sys, math, collections, statistics as st
from rank_bm25 import BM25Okapi

ROOT = os.path.join(os.path.dirname(__file__), "..")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "lite"
RES = os.path.join(ROOT, "results", SPLIT); os.makedirs(RES, exist_ok=True)
tasks = [json.loads(l) for l in open(os.path.join(ROOT, f"data/tasks_{SPLIT}.jsonl"))]
def tree(t):
    return json.load(open(os.path.join(ROOT, "data/trees", f"{t['repo'].replace('/','__')}__{t['base_commit']}.json")))

TOK_PER_BYTE = 1/3.5            # heuristic for Python source (stated in paper)
def toks(nbytes): return nbytes * TOK_PER_BYTE

def split_ident(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", s) if len(w) > 1]

def is_test(p): return bool(re.search(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*$|_tests?\.py$|conftest\.py$", p))
def is_doc(p): return bool(re.search(r"(^|/)(docs?|doc|examples?|benchmarks?|asv_bench)(/|$)", p))

# ---------- signal extraction ----------
PATH_RE = re.compile(r"[\w./\\-]+\.py\b")
TB_RE = re.compile(r'File "([^"]+\.py)", line \d+')
DOTTED_RE = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*){1,})\b")
IDENT_RE = re.compile(r"`([^`\n]{2,80})`|\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[a-z]+(?:_[a-z0-9]+)+)\b")

def suffix_match(mention, paths):
    m = mention.replace("\\", "/").lstrip("./")
    parts = m.split("/")
    for k in range(len(parts), 0, -1):          # longest suffix that resolves
        suf = "/".join(parts[-k:])
        hits = [p for p in paths if p == suf or p.endswith("/" + suf)]
        if hits and (k >= 2 or len(hits) == 1): return hits
    return []

def explicit_mentions(text, paths):
    """Files named by path, traceback frame, or dotted module path. Returns (ordered list, kinds)."""
    out, kinds = [], set()
    tb = TB_RE.findall(text)
    for f in reversed(tb):                       # innermost frame first
        for h in suffix_match(f, paths):
            if h not in out: out.append(h); kinds.add("traceback")
    for f in PATH_RE.findall(text):
        for h in suffix_match(f, paths):
            if h not in out: out.append(h); kinds.add("path")
    modset = {p[:-3].replace("/", "."): p for p in paths if p.endswith(".py")}
    modset.update({p[:-12].replace("/", "."): p for p in paths if p.endswith("/__init__.py")})
    for d in DOTTED_RE.findall(text):            # longest importable prefix, e.g. django.db.models.query.QuerySet
        segs = d.split(".")
        for k in range(len(segs), 1, -1):
            key = ".".join(segs[:k])
            cand = [p for m, p in modset.items() if m == key or m.endswith("." + key)]
            if len(cand) == 1 or (cand and k >= 3):
                for h in cand[:1]:
                    if h not in out: out.append(h); kinds.add("module")
                break
    return out, kinds

# ---------- rankers ----------
def candidates(paths, prior):
    py = [p for p in paths if p.endswith(".py")]
    return [p for p in py if not (is_test(p) or is_doc(p))] if prior else py

def bm25_rank(text, cands):
    docs = [split_ident(p) + split_ident(os.path.basename(p)[:-3]) * 2 for p in cands]
    bm = BM25Okapi(docs)
    sc = bm.get_scores(split_ident(text))
    return [p for _, p in sorted(zip(sc, cands), key=lambda x: -x[0])]

def ident_rank(text, cands):
    ids = collections.Counter()
    for a, b in IDENT_RE.findall(text):
        for w in split_ident(a or b): ids[w] += 1
    score = {}
    for p in cands:
        comp = split_ident(p)
        stem = set(split_ident(os.path.basename(p)[:-3]))
        s = sum(math.log1p(c) * (3 if w in stem else 1) for w, c in ids.items() if w in comp)
        score[p] = s
    return sorted(cands, key=lambda p: -score[p])

def rrf(*ranks, k=60):
    s = collections.defaultdict(float)
    for r in ranks:
        for i, p in enumerate(r): s[p] += 1 / (k + i)
    return sorted(s, key=lambda p: -s[p])

def signalrank(text, paths, prior=True, use_explicit=True, fuse_ident=False):
    c = candidates(paths, prior)
    fused = rrf(bm25_rank(text, c), ident_rank(text, c)) if fuse_ident else bm25_rank(text, c)
    if not use_explicit: return fused
    ex, _ = explicit_mentions(text, paths)
    ex = [p for p in ex if p in set(c)] + [p for p in ex if p not in set(c)]
    return ex + [p for p in fused if p not in set(ex)]

# ---------- evaluation ----------
KS = [1, 3, 5, 10]
BUDGETS = [4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
def evaluate(name, fn):
    hit = {k: 0 for k in KS}; allk = {k: 0 for k in KS}; bud = {b: 0 for b in BUDGETS}; rr = []
    tok_to_hit = []; firsts = []
    for t in tasks:
        files = tree(t); size = {f["path"]: f["size"] for f in files}
        r = fn(t["problem_statement"], list(size)); g = set(t["gold_files"])
        pos = [i for i, p in enumerate(r) if p in g]
        first = pos[0] if pos else None; firsts.append(first if first is not None else math.inf)
        rr.append(1 / (first + 1) if first is not None else 0)
        for k in KS:
            hit[k] += first is not None and first < k
            allk[k] += g <= set(r[:k])
        cum = 0; t2h = math.inf
        for i, p in enumerate(r):
            cum += toks(size[p])
            if p in g: t2h = cum; break
        tok_to_hit.append(t2h)
        for b in BUDGETS: bud[b] += t2h <= b
    n = len(tasks)
    fin = [x for x in tok_to_hit if x < math.inf]
    return {"name": name, **{f"hit@{k}": hit[k]/n for k in KS}, **{f"all@{k}": allk[k]/n for k in KS},
            "mrr": sum(rr)/n, **{f"budget@{b//1000}k": bud[b]/n for b in BUDGETS},
            "median_tokens_to_first_hit": st.median(fin) if fin else None, "_t2h": tok_to_hit, "_first": firsts}

import random
def random_rank(text, paths):
    c = candidates(paths, True); import zlib; rnd = random.Random(zlib.crc32(text.encode())); rnd.shuffle(c); return c

runs = [
  ("Random (src .py)",            random_rank),
  ("BM25 on paths",               lambda t,p: bm25_rank(t, candidates(p, False))),
  ("BM25 on paths + src prior",   lambda t,p: bm25_rank(t, candidates(p, True))),
  ("Identifier match + prior",    lambda t,p: ident_rank(t, candidates(p, True))),
  ("RRF(BM25, ident) + prior",    lambda t,p: signalrank(t, p, True, False, True)),
  ("Explicit -> RRF(BM25, ident)",lambda t,p: signalrank(t, p, True, True, True)),
  ("SignalRank w/o prior",        lambda t,p: signalrank(t, p, False, True)),
  ("SignalRank (full)",           lambda t,p: signalrank(t, p, True, True)),
]
results = [evaluate(n, f) for n, f in runs]
for r in results:
    print(f"{r['name']:30s} " + " ".join(f"{k}={r[k]:.3f}" for k in ["hit@1","hit@5","hit@10","all@10","mrr","budget@8k","budget@32k","budget@128k"]))

# ---------- paired bootstrap: SignalRank vs strongest single baseline ----------
R = {r["name"]: r for r in results}; rng = random.Random(0); boot = {}
for k in (1, 5, 10):
    a = [x < k for x in R["SignalRank (full)"]["_first"]]; b = [x < k for x in R["BM25 on paths + src prior"]["_first"]]
    d = []
    for _ in range(10000):
        idx = [rng.randrange(len(a)) for _ in a]; d.append(sum(a[i] - b[i] for i in idx) / len(a))
    d.sort(); boot[f"hit@{k}"] = {"delta": (sum(a) - sum(b)) / len(a), "ci95": [d[250], d[9749]]}
print("bootstrap SignalRank - BM25+prior:", json.dumps(boot))

# ---------- descriptive analysis ----------
desc = collections.Counter(); strata = collections.defaultdict(lambda: [0, 0]); per_repo = collections.defaultdict(lambda: [0, 0])
gold_tok, repo_tok, ngold, gold_in_explicit = [], [], [], 0
full = dict(runs)["SignalRank (full)"]
for t in tasks:
    files = tree(t); size = {f["path"]: f["size"] for f in files}; g = set(t["gold_files"])
    ex, kinds = explicit_mentions(t["problem_statement"], list(size))
    stratum = "traceback" if "traceback" in kinds else "path/module" if kinds else "no explicit file signal"
    desc[stratum] += 1
    top1 = full(t["problem_statement"], list(size))[0] in g
    strata[stratum][0] += top1; strata[stratum][1] += 1
    per_repo[t["repo"]][0] += top1; per_repo[t["repo"]][1] += 1
    gold_in_explicit += bool(g & set(ex))
    gold_tok += [toks(size[p]) for p in g if p in size]
    repo_tok.append(toks(sum(s for p, s in size.items() if p.endswith(".py"))))
    ngold.append(len(g))
summary = {
  "n_tasks": len(tasks), "n_repos": len({t['repo'] for t in tasks}),
  "multi_file_gold_frac": sum(x > 1 for x in ngold)/len(ngold), "single_file_gold_frac": sum(x == 1 for x in ngold)/len(ngold),
  "median_repo_py_tokens": st.median(repo_tok), "min_repo_py_tokens": min(repo_tok), "max_repo_py_tokens": max(repo_tok),
  "median_gold_file_tokens": st.median(gold_tok),
  "gold_file_tokens_p90": sorted(gold_tok)[int(.9*len(gold_tok))],
  "frac_gold_files_over_8k": sum(x > 8000 for x in gold_tok)/len(gold_tok),
  "frac_gold_files_over_32k": sum(x > 32000 for x in gold_tok)/len(gold_tok),
  "signal_strata": dict(desc),
  "top1_by_stratum": {k: v[0]/v[1] for k, v in strata.items()},
  "frac_issue_explicitly_names_a_gold_file": gold_in_explicit/len(tasks),
  "top1_by_repo": {k: [v[0], v[1]] for k, v in sorted(per_repo.items())},
}
print(json.dumps(summary, indent=1))
json.dump({"results": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results], "bootstrap_vs_bm25_prior": boot, "summary": summary},
          open(os.path.join(RES, "results.json"), "w"), indent=1)

# ---------- figure: hit-rate vs token budget ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = [2**i * 1000 for i in range(0, 9)]
fig, ax = plt.subplots(figsize=(6, 3.6))
for r in results:
    if r["name"] in ("Random (src .py)", "BM25 on paths", "BM25 on paths + src prior", "SignalRank (full)"):
        ax.plot([x/1000 for x in xs], [sum(t <= x for t in r["_t2h"])/len(tasks) for x in xs], marker="o", label=r["name"])
ax.set_xscale("log", base=2); ax.set_xlabel("Context budget for whole-file reading (K tokens)")
ax.set_ylabel("Tasks with ≥1 gold file in budget"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "budget_curve.png"), dpi=200)
