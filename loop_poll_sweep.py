"""Bounded 15s poller: checks for the sweep JSON, then EXITS so the harness notifies.
An infinite poller never completes and therefore never pings - that was the bug.
Usage: python loop_poll_sweep.py [max_polls]
"""
import json
import sys
import time
from pathlib import Path

OUT = Path("kaggle_out_probe/threshold_sweep.json")
LOG = Path("C:/Users/Dell/.local/share/opencode/shell/90eb137e72a69dff444812a1d8c71f7816f9836d/sh_158f9dd80001sMwcqpaaqb4dix.out")
max_polls = int(sys.argv[1]) if len(sys.argv) > 1 else 12

for n in range(1, max_polls + 1):
    if OUT.exists() and OUT.stat().st_size > 100:
        print(f"LANDED at poll {n} ({n*15}s)", flush=True)
        d = json.loads(OUT.read_text())
        print("MATCHED:", json.dumps(d.get("matched"), indent=1), flush=True)
        print("TARGET:", d.get("target_malignant"), flush=True)
        sys.exit(0)
    tail = ""
    if LOG.exists():
        t = LOG.read_text(errors="replace").strip().splitlines()
        tail = t[-1] if t else ""
    print(f"poll {n}/{max_polls} ({n*15}s) | sweep: {tail[:70]}", flush=True)
    time.sleep(15)

print("STILL RUNNING - re-arm", flush=True)
sys.exit(1)