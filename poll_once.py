"""Single-cycle poller: kernel status + live GPU quota burn. Appends to poll_log.txt.

Runs in well under a second and is safe to invoke from a Windows Scheduled Task
every 15 minutes, because it holds no state and never sleeps.

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
SLUG = sys.argv[1] if len(sys.argv) > 1 else "juyhtrfewqasdfrgt/bone-cancer-v27-train"

PROBE = (
    "import os,json;from pathlib import Path;"
    "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
    "from kaggle.api.kaggle_api_extended import KaggleApi;"
    "a=KaggleApi();a.authenticate();"
    "s=a.kernels_status('{slug}');"
    "d=json.loads(a.quota_view().to_json());"
    "g=d['gpuQuota'];g=json.loads(g) if isinstance(g,str) else g;"
    "n=lambda x: float(str(x).rstrip('s').split('.')[0]);"
    "print(json.dumps({{'status':str(s.status),'fail':(s.failureMessage or '')[:160],"
    "'used':n(g['timeUsed']),'left':(108000-n(g['timeUsed']))/3600,"
    "'reserved':n(g['timeReserved'])/3600}}))"
).format(slug=SLUG)

r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd=ROOT)
line = (r.stdout or r.stderr).strip()
stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
entry = f"{stamp} | {line}"
print(entry, flush=True)
with LOG.open("a", encoding="utf-8") as f:
    f.write(entry + "\n")

try:
    d = json.loads(line)
    if "COMPLETE" in d["status"] or "ERROR" in d["status"]:
        (ROOT / "poll_final_state.txt").write_text(f"{d['status']}\n{d.get('fail','')}\n")
except Exception:
    pass
