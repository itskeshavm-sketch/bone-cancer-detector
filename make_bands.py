"""Turn the measured calibration into the band table the demo uses.

Reads checkpoints/uncertainty_calibration.json (564 test films, single-view path)
and writes checkpoints/uncertainty_bands.json:
  - confidence edges -> measured accuracy on real films in that band
  - a "needs radiologist" flag for bands whose measured error rate is at least
    UNCERTAIN_ERR (3x the model's overall error rate)

Why a table instead of one threshold: the sweep shows error rate falls off a cliff
above ~0.90 confidence (35% error below vs 6.7% above), so the band boundaries
follow the data rather than an arbitrary cutoff.
"""
import json
from pathlib import Path

UNCERTAIN_ERR = 0.25  # 3x the ~11.7% overall single-view error rate
EDGES = [0.0, 0.60, 0.70, 0.80, 0.90, 1.01]

cal = json.loads(Path("checkpoints/uncertainty_calibration.json").read_text())
rows = cal["rows"]
n = len(rows)
overall_acc = cal["single_view_acc"]
overall_err = 1 - overall_acc

bands = []
for i in range(len(EDGES) - 1):
    lo, hi = EDGES[i], EDGES[i + 1]
    sel = [r for r in rows if lo <= r["top_conf"] < hi]
    if not sel:
        bands.append({"lo": lo, "hi": hi, "n": 0, "accuracy": overall_acc,
                      "error_rate": overall_err, "uncertain": False})
        continue
    acc = sum(r["correct"] for r in sel) / len(sel)
    err = 1 - acc
    bands.append({
        "lo": lo, "hi": hi, "n": len(sel),
        "accuracy": acc, "error_rate": err,
        "share": len(sel) / n,
        "uncertain": err >= UNCERTAIN_ERR,
    })

# coverage: how many of all errors fall in uncertain bands
err_total = sum(1 for r in rows if not r["correct"])
flagged = [r for r in rows if any(
    b["uncertain"] and b["lo"] <= r["top_conf"] < b["hi"] for b in bands)]
flagged_err = sum(1 for r in flagged if not r["correct"])

out = {
    "model": "V12 (stageA12 gate + stageB12 specialist)",
    "n_films": n,
    "overall_accuracy": overall_acc,
    "overall_error_rate": overall_err,
    "uncertain_error_threshold": UNCERTAIN_ERR,
    "bands": bands,
    "flagged_count": len(flagged),
    "flagged_share": len(flagged) / n,
    "flagged_error_rate": (flagged_err / len(flagged)) if flagged else 0.0,
    "errors_total": err_total,
    "errors_caught": flagged_err,
    "error_capture": flagged_err / err_total if err_total else 0.0,
    "reliable_band": next((b for b in bands if not b["uncertain"] and b["n"] > 0), None),
}
Path("checkpoints/uncertainty_bands.json").write_text(json.dumps(out, indent=1))

print(f"overall: {n} films, acc {overall_acc:.4f}, err {overall_err:.4f}")
print(f"{'band':<16}{'n':>5}{'acc':>9}{'err':>8}{'share':>8}  uncertain")
for b in bands:
    print(f"[{b['lo']:.2f},{b['hi']:.2f})   {b['n']:>5}{b['accuracy']:>9.3f}"
          f"{b['error_rate']:>8.3f}{b.get('share', 0):>8.1%}  "
          f"{'YES' if b['uncertain'] else 'no'}")
print(f"\nuncertain band: {len(flagged)} films ({len(flagged)/n:.1%} of all), "
      f"error rate {out['flagged_error_rate']:.1%}")
print(f"catches {flagged_err}/{err_total} errors = {out['error_capture']:.1%}")
print("wrote checkpoints/uncertainty_bands.json")