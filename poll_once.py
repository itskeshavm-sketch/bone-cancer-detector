"""Single-cycle poller: kernel status + live GPU quota. Appends to poll_log.txt.

Runs in about a second and holds no state, so it is safe to invoke from the
Windows Scheduled Task every 15 minutes.

The quota parser reads `totalTimeAllowed` from the API instead of assuming
30 hours. That assumption was wrong: the API reports 21600s (6 h) per week for
this account, and hardcoding 108000 made the remaining time look 4x larger than
it is. Quota accounting is the thing that decides how many more training runs
are possible, so it is read, not assumed.

Usage: python poll_once.py [kernel_slug]
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector")
LOG = ROOT / "poll_log.txt"
SLUG = sys.argv[1] if len(sys.argv) > 1 else "juyhtrfewqasdfrgt/bone-cancer-v29-train"

PROBE = """
import os, json
from pathlib import Path
os.environ['KAGGLE_API_TOKEN'] = (Path.home() / '.kaggle' / 'access_token').read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi


def secs(x):
    t = str(x).rstrip('s')
    parts = t.split(':')
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(t.split('.')[0])


a = KaggleApi()
a.authenticate()
s = a.kernels_status('SLUG')
d = json.loads(a.quota_view().to_json())
g = d['gpuQuota']
used = secs(g['timeUsed'])
total = secs(g['totalTimeAllowed'])
print(json.dumps({
    'status': str(s.status),
    'fail': (s.failureMessage or '')[:160],
    'used_h': round(used / 3600, 3),
    'total_h': round(total / 3600, 2),
    'left_h': round((total - used) / 3600, 3),
    'reserved_h': round(secs(g['timeReserved']) / 3600, 3),
    'refresh': d.get('quotaRefreshTime'),
}))
""".replace("SLUG", SLUG)

r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd=ROOT)
line = (r.stdout or r.stderr).strip()
stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
entry = f"{stamp} | {line}"
print(entry, flush=True)
try:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(entry + "\n")
except Exception:
    pass

try:
    d = json.loads(line)
    if "COMPLETE" in d.get("status", "") or "ERROR" in d.get("status", ""):
        (ROOT / "poll_final_state.txt").write_text(
            f"{d['status']}\n{d.get('fail', '')}\n")
except Exception:
    pass
