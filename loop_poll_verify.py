"""Bounded 15s poller for the verify job. Exits on completion so the harness pings.
Usage: python loop_poll_verify.py [max_polls]
"""
import json
import sys
import time
from pathlib import Path

DONE = Path("kaggle_out_probe/verify_done.flag")
BANDS = Path("checkpoints/uncertainty_bands_v21.json")
PROG = Path("kaggle_out_probe/verify_progress.txt")
max_polls = int(sys.argv[1]) if len(sys.argv) > 1 else 20

for n in range(1, max_polls + 1):
    if DONE.exists() and BANDS.exists():
        print(f"LANDED at poll {n} ({n*15}s)", flush=True)
        d = json.loads(BANDS.read_text())
        print("BENCH:", json.dumps(d.get("benchmark_single_view"), indent=1), flush=True)
        print("BANDS:", json.dumps([{k: b[k] for k in ("lo", "hi", "n", "accuracy", "uncertain")}
                                    for b in d["bands"]], indent=1), flush=True)
        print("FLAGGED:", d["flagged_share"], "capture:", d["error_capture"], flush=True)
        sys.exit(0)
    tail = ""
    if PROG.exists():
        t = PROG.read_text(errors="replace").strip().splitlines()
        tail = t[-1] if t else ""
    print(f"poll {n}/{max_polls} ({n*15}s) | {tail[:90]}", flush=True)
    time.sleep(15)

print("STILL RUNNING - re-arm", flush=True)
sys.exit(1)