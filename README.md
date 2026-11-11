# SignalRank: budgeted, LLM-free bug localization for small-context coding agents

Code for the paper "Ranking Is Not the Bottleneck" (Gemma 4 Developer Agent, Paper Track).

```bash
pip install -r requirements.txt
gh auth login                     # GitHub tree API calls need authentication (~1,000 calls)
python code/fetch_data.py lite    # SWE-bench Lite, 300 tasks
python code/fetch_data.py heldout # SWE-bench Verified minus Lite, 407 tasks
python code/localize.py lite
python code/localize.py heldout   # writes results/<split>/results.json and budget_curve.png
```
Deterministic; runs on a laptop CPU in under 2 minutes once the data is fetched.
