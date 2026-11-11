"""Step 1: download SWE-bench Lite (test, 300 tasks) and the full file tree
(paths + byte sizes) of each repo at its base commit via the GitHub API."""
import json, os, re, subprocess, concurrent.futures as cf
import sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
DS = sys.argv[1] if len(sys.argv) > 1 else "lite"
if DS == "lite":
    df = pd.read_parquet(hf_hub_download("princeton-nlp/SWE-bench_Lite", "data/test-00000-of-00001.parquet", repo_type="dataset"))
else:  # held-out: SWE-bench Verified minus any instance that is also in Lite
    lite = pd.read_parquet(hf_hub_download("princeton-nlp/SWE-bench_Lite", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    df = pd.read_parquet(hf_hub_download("princeton-nlp/SWE-bench_Verified", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    df = df[~df.instance_id.isin(lite.instance_id)].reset_index(drop=True)
df["gold_files"] = df.patch.apply(lambda s: sorted(set(re.findall(r"^diff --git a/(\S+) b/", s, re.M))))
df[["instance_id","repo","base_commit","problem_statement","gold_files","created_at"]].to_json(
    os.path.join(OUT, f"tasks_{DS}.jsonl"), orient="records", lines=True)

tree_dir = os.path.join(OUT, "trees"); os.makedirs(tree_dir, exist_ok=True)
def fetch(row):
    fn = os.path.join(tree_dir, f"{row.repo.replace('/','__')}__{row.base_commit}.json")
    if os.path.exists(fn): return
    r = subprocess.run(["gh","api",f"repos/{row.repo}/git/trees/{row.base_commit}?recursive=1"],capture_output=True,text=True)
    t = json.loads(r.stdout)
    assert not t.get("truncated"), row.instance_id
    files = [{"path":e["path"],"size":e["size"]} for e in t["tree"] if e["type"]=="blob"]
    json.dump(files, open(fn,"w"))
with cf.ThreadPoolExecutor(8) as ex:
    list(ex.map(fetch, [r for _,r in df.drop_duplicates(["repo","base_commit"]).iterrows()]))
print(len(df), "tasks;", len(os.listdir(tree_dir)), "trees")
