"""Bounded poller for the V22 kernel: status + live GPU quota burn. Exits so the
harness notifies. Usage: python loop_poll_v22.py [max_polls]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector"
SLUG = "juyhtrfewqasdfrgt/bone-cancer-v22-train"
max_polls = int(sys.argv[1]) if len(sys.argv) > 1 else 4

print(f"poll-sleep 900s (watching {SLUG})", flush=True)
time.sleep(900)

for n in range(max_polls):
    r = subprocess.run([sys.executable, "-c",
                        "import os;from pathlib import Path;"
                        "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
                        "from kaggle.api.kaggle_api_extended import KaggleApi;"
                        "a=KaggleApi();a.authenticate();"
                        f"s=a.kernels_status('{SLUG}');"
                        "print(s.status,'|',(s.failureMessage or '')[:200])"],
                       capture_output=True, text=True, cwd=ROOT)
    print(f"POLL {n+1}/{max_polls}: {(r.stdout or r.stderr).strip()[:300]}", flush=True)
    q = subprocess.run([sys.executable, "-c",
                        "import os,json;from pathlib import Path;"
                        "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
                        "from kaggle.api.kaggle_api_extended import KaggleApi;"
                        "a=KaggleApi();a.authenticate();"
                        "d=json.loads(a.quota_view().to_json());"
                        "g=d['gpuQuota'];g=json.loads(g) if isinstance(g,str) else g;"
                        "n=lambda s: float(str(s).rstrip('s').split('.')[0]);"
                        "print(json.dumps({'used':n(g['timeUsed']),'total':n(g['totalTimeAllowed'])}))"],
                       capture_output=True, text=True, cwd=ROOT)
    if q.stdout.strip():
        d = json.loads(q.stdout.strip())
        print(f"QUOTA gpu_used={d['used']/3600:.2f}h weekly_left={(108000-d['used'])/3600:.2f}h",
              flush=True)
    if "COMPLETE" in (r.stdout or "") or "ERROR" in (r.stdout or ""):
        print("FINAL STATE REACHED", flush=True)
        sys.exit(0)
    if n + 1 < max_polls:
        time.sleep(900)

print("still running - re-arm", flush=True)
sys.exit(1)