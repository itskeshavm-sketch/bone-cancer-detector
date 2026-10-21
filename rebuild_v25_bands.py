"""Re-derive V25's band table at the SHIPPED operating point (mal 0.30).

The cached per-film rows already hold every probability, so this is a pure
re-grouping - no inference, no re-download. The band table has to be measured
at the threshold the demo actually uses, otherwise the "accuracy on confident
reads" figure describes a different model configuration.
"""
import json
from pathlib import Path

GATE_THR, MAL_THR = 0.30, 0.30
UNCERTAIN_ERR = 0.25
EDGES = [0.0, 0.60, 0.70, 0.80, 0.90, 1.01]

rows = json.loads(Path("checkpoints/benchmark_rows_v25.json").read_text())
n = len(rows)
per = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
scored = []
for r in rows:
    if r["tumor"] < GATE_THR:
        pred, top = 2, 1.0 - r["tumor"]
    else:
        pred = 1 if r["mal"] >= MAL_THR else 0
        top = max(r["mal"], r["ben"])
    good = int(pred == r["truth"])
    per[r["truth"]][0] += good
    per[r["truth"]][1] += 1
    scored.append({**r, "pred": pred, "top_conf": top, "correct": good})

ok = sum(r["correct"] for r in scored)
print(f"benchmark single-view at gate {GATE_THR} / malignant {MAL_THR}: "
      f"{ok}/{n} = {ok/n:.4f}")
print(f"  benign {per[0][0]}/{per[0][1]} | malignant {per[1][0]}/{per[1][1]} | "
      f"normal {per[2][0]}/{per[2][1]}")

bands = []
for i in range(len(EDGES) - 1):
    lo, hi = EDGES[i], EDGES[i + 1]
    sel = [r for r in scored if lo <= r["top_conf"] < hi]
    if not sel:
        bands.append({"lo": lo, "hi": hi, "n": 0, "accuracy": ok / n,
                      "error_rate": 1 - ok / n, "uncertain": False})
        continue
    acc = sum(r["correct"] for r in sel) / len(sel)
    bands.append({"lo": lo, "hi": hi, "n": len(sel), "accuracy": acc,
                  "error_rate": 1 - acc, "share": len(sel) / n,
                  "uncertain": (1 - acc) >= UNCERTAIN_ERR})

err_total = sum(1 for r in scored if not r["correct"])
flagged = [r for r in scored if any(
    b["uncertain"] and b["lo"] <= r["top_conf"] < b["hi"] for b in bands)]
flagged_err = sum(1 for r in flagged if not r["correct"])

out = {
    "model": f"V25 (stageA18 gate + stageB25 specialist, gate {GATE_THR}, "
             f"malignant {MAL_THR})",
    "n_films": n, "overall_accuracy": ok / n, "overall_error_rate": 1 - ok / n,
    "uncertain_error_threshold": UNCERTAIN_ERR, "bands": bands,
    "flagged_count": len(flagged), "flagged_share": len(flagged) / n,
    "flagged_error_rate": (flagged_err / len(flagged)) if flagged else 0.0,
    "errors_total": err_total, "errors_caught": flagged_err,
    "error_capture": flagged_err / err_total if err_total else 0.0,
    "reliable_band": next((b for b in bands if not b["uncertain"] and b["n"] > 0), None),
    "per_class": {str(k): f"{v[0]}/{v[1]}" for k, v in per.items()},
}
Path("checkpoints/uncertainty_bands_v25.json").write_text(json.dumps(out, indent=1))
Path("checkpoints/benchmark_rows_v25.json").write_text(json.dumps(scored, indent=1))

print(f"\n{'band':<16}{'n':>5}{'acc':>9}{'err':>8}  uncertain")
for b in bands:
    print(f"[{b['lo']:.2f},{b['hi']:.2f})   {b['n']:>5}{b['accuracy']:>9.3f}"
          f"{b['error_rate']:>8.3f}  {'YES' if b['uncertain'] else 'no'}")
print(f"uncertain band: {len(flagged)} films ({len(flagged)/n:.1%}), err "
      f"{out['flagged_error_rate']:.1%}, captures {flagged_err}/{err_total} = "
      f"{out['error_capture']:.1%}")
print("\nwrote checkpoints/uncertainty_bands_v25.json")