"""15-minute progress ping for the harvest.

Reports the harvested-radiograph count and the log tail, then EXITS so the harness
notifies. A multi-cycle poller inside one process would silence me for the whole
hour it covers - that was the bug that made pings feel hourly.

Usage: python loop_poll_harvest.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector")
LOG = ROOT / "kaggle_out_probe" / "harvest_bt_wide.log"

print("poll-sleep 900s...", flush=True)
time.sleep(900)

bt = ROOT / "wild_train2"
wm = ROOT / "wild_train3"
n_bt = len(list(bt.glob("bt2_*.png"))) if bt.exists() else 0
n_wm = len(list(wm.glob("wm_*.png"))) if wm.exists() else 0
print(f"HARVEST bonetumor={n_bt} wikimedia={n_wm} total={n_bt + n_wm}", flush=True)

if LOG.exists():
    lines = LOG.read_text(errors="replace").strip().splitlines()
    print("LOG TAIL:", flush=True)
    for l in lines[-6:]:
        print("  " + l[:110], flush=True)

# quota + kernel status, so one ping covers everything
for label, cmd in [
    ("QUOTA", "import os,json;from pathlib import Path;"
              "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
              "from kaggle.api.kaggle_api_extended import KaggleApi;a=KaggleApi();a.authenticate();"
              "d=json.loads(a.quota_view().to_json());g=d['gpuQuota'];"
              "g=json.loads(g) if isinstance(g,str) else g;"
              "n=lambda s: float(str(s).rstrip('s').split('.')[0]);"
              "print(f\"gpu_used={n(g['timeUsed'])/3600:.2f}h weekly_left={(108000-n(g['timeUsed']))/3600:.2f}h\")"),
]:
    r = subprocess.run([sys.executable, "-c", cmd], capture_output=True, text=True, cwd=ROOT)
    print(f"{label}: {(r.stdout or r.stderr).strip()[:200]}", flush=True)