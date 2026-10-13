"""
Print non-PASS results with their SQL trace, and a per-family summary.

    PYTHONPATH=. uv run python experiments/bulk_eval/inspect_results.py [results.jsonl] [--summary] [--family F] [--verdict V]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default=str(HERE / "results.jsonl"))
ap.add_argument("--summary", action="store_true")
ap.add_argument("--family")
ap.add_argument("--verdict")
args = ap.parse_args()

Q = {x["id"]: x for x in json.loads((HERE / "questions.json").read_text(encoding="utf-8"))}
R = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines() if l.strip()]

if args.summary:
    amb = [r for r in R if Q[r["id"]].get("ambiguous")]
    R = [r for r in R if not Q[r["id"]].get("ambiguous")]
    print(f"(excluding {len(amb)} questions marked ambiguous in the bank)")
    total = Counter(r["verdict"] for r in R)
    print(f"{len(R)} results: " + ", ".join(f"{k}={v}" for k, v in total.most_common()))
    fam = defaultdict(Counter)
    for r in R:
        fam[Q[r["id"]]["family"]][r["verdict"]] += 1
    rows = sorted(fam.items(), key=lambda kv: -(sum(kv[1].values()) - kv[1]["PASS"]) / sum(kv[1].values()))
    for f, c in rows:
        n = sum(c.values())
        print(f"  {f:<26} {c['PASS']:>4}/{n:<4} pass  " + " ".join(f"{k}={v}" for k, v in c.items() if k != "PASS"))
    sys.exit()

for r in R:
    x = Q[r["id"]]
    if r["verdict"] == "PASS" or (args.family and x["family"] != args.family) or (args.verdict and r["verdict"] != args.verdict):
        continue
    print(f"===== {r['id']} [{r['verdict']}] ({x['family']}) {x['question']}")
    print(f"  EXPECT {x['mode']} {x['expect']} {x.get('any_of', '')}")
    for t in r["trace"]:
        if "SQL agent start" in t:
            print("  SUB-Q:", t.split("SQL agent start: ", 1)[-1][:300])
        elif "GENERATED_SQL" in t:
            print("  SQL:  ", " ".join(t.split()[1:])[:600])
        elif "SQL_RESULT" in t:
            print("  RES:  ", " | ".join(t.splitlines()[1:])[:300])
        else:
            print("  LOG:  ", t.replace("\n", " ")[:300])
    if r.get("exception") or r.get("timeout"):
        print("  EXC:  ", r.get("exception") or "timeout")
    print("  ANSWER:", (r.get("answer") or "").replace("\n", " ")[:400])
