"""Launch the V24 band calibration DETACHED, so a server restart cannot kill it.

Runs calibrate_v24_bands.py in its own process (Start-Process), which survives
the opencode server restarting. The calibration itself is resumable: it writes
per-film rows to disk as it goes, and a re-run picks up where it stopped.

Usage: python launch_calibrate.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LOG = ROOT / "kaggle_out_probe" / "calibrate_v24.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

# make the calibration itself resumable: keep rows that already exist
src = ROOT / "calibrate_v24_bands.py"
code = src.read_text(encoding="utf-8")
if "RESUME" not in code:
    patched = code.replace(
        'rows = []\nfor cls in ["benign", "malignant", "normal"]:',
        '# RESUME: reuse rows already computed in a previous (possibly killed) run\n'
        'ROWS_PATH = Path("checkpoints/benchmark_rows_v24.json")\n'
        'rows = json.loads(ROWS_PATH.read_text()) if ROWS_PATH.exists() else []\n'
        'done_files = {r["file"] for r in rows}\n'
        'print(f"resuming with {len(rows)} rows", flush=True)\n'
        'for cls in ["benign", "malignant", "normal"]:')
    patched = patched.replace(
        '    files = sorted(Path(f"data/test/{cls}").glob("*"))\n'
        '    for start in range(0, len(files), BATCH):',
        '    files = [p for p in sorted(Path(f"data/test/{cls}").glob("*"))\n'
        '            if str(p) not in done_files]\n'
        '    if not files:\n'
        '        continue\n'
        '    for start in range(0, len(files), BATCH):')
    patched = patched.replace(
        '    print(f"  {cls} done ({len(rows)} rows)", flush=True)',
        '    print(f"  {cls} done ({len(rows)} rows)", flush=True)\n'
        '    ROWS_PATH.write_text(json.dumps(rows, indent=1))')
    tmp = ROOT / "kaggle_out_probe" / "_calibrate_v24_resumable.py"
    tmp.write_text(patched, encoding="utf-8")
    print("wrote resumable copy:", tmp)
else:
    tmp = src

with open(LOG, "ab") as lf:
    p = subprocess.Popen([sys.executable, "-u", str(tmp)], cwd=str(ROOT),
                         stdout=lf, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                         | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
print("launched detached PID", p.pid, "-> log", LOG)