"""Self-re-arming 15-minute poller for a Kaggle kernel.

Unlike the earlier one-shot pollers, this one writes a heartbeat file every cycle
and holds a lock file, so the Windows Scheduled Task that restarts it after a
server restart does not spawn duplicate loops.

Exits (and therefore notifies the harness) when the kernel reaches a final state.
With a cycle count it exits after that many cycles so the harness is pinged once
per cycle and the agent can re-arm it.

Usage: python loop_poll_v27.py [slug] [cycles]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector")
SLUG = sys.argv[1] if len(sys.argv) > 1 else "juyhtrfewqasdfrgt/bone-cancer-v28-train"
CYCLES = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 = run until final state
# The scheduled task and the harness-level notifier run the SAME kernel, so they
# need SEPARATE lock files - otherwise the second one exits instantly thinking
# the first is already covering it, and the harness never gets pinged.
TAG = sys.argv[3] if len(sys.argv) > 3 else "sched"
LOCK = ROOT / f"poll_{TAG}.lock"
BEAT = ROOT / f"poll_heartbeat_{TAG}.txt"
CYCLE = 900


def alive():
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text().strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if alive():
    print("poller already running - exiting", flush=True)
    sys.exit(2)

LOCK.write_text(str(os.getpid()))

# Write the status line to the file directly rather than relying on stdout
# redirection: Start-Process -RedirectStandardOutput makes the parent shell wait
# on the child's handles, which hung an interactive invocation and could wedge
# the scheduled task.
STATUS = ROOT / "poll_loop.out"


def note(msg):
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


note(f"poller armed pid={os.getpid()} slug={SLUG} cycle={CYCLE}s")

try:
    done = 0
    while True:
        # Sleep FIRST, then poll. The harness-level poller must not report
        # immediately: the point of it is to be woken up 15 minutes later.
        time.sleep(CYCLE)
        r = subprocess.run([sys.executable, str(ROOT / "poll_once.py"), SLUG],
                           capture_output=True, text=True, cwd=ROOT)
        out = (r.stdout or r.stderr).strip()
        note(f"CYCLE {out}")
        BEAT.write_text(out + "\n")
        if "COMPLETE" in out or "ERROR" in out:
            note("FINAL STATE REACHED - notifier exits")
            sys.exit(0)
        done += 1
        if CYCLES and done >= CYCLES:
            note("cycle budget spent - re-arm me")
            sys.exit(1)
finally:
    try:
        LOCK.unlink()
    except OSError:
        pass
