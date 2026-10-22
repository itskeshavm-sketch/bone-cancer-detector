# Launch the 15-minute poller as a detached process.
# Invoked by the Windows Scheduled Task every 15 minutes.
#
# Two things this must get right:
#  1. Exactly ONE poller per tag. A server restart leaves the lock file behind
#     with a dead pid, and the naive "is the pid alive" check then spawns a
#     second one - which is how seven pollers ended up hammering the Kaggle
#     API at once. So before starting, every existing poller for this target is
#     killed, and the lock is rebuilt from scratch.
#  2. The target kernel is read from poll_target.txt so the task never needs
#     editing.

$root = "C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector"
$slugFile = "$root\poll_target.txt"
if (Test-Path $slugFile) {
    $slug = (Get-Content $slugFile -Raw).Trim()
} else {
    $slug = "juyhtrfewqasdfrgt/bone-cancer-v29-train"
}
if (-not $slug) { $slug = "juyhtrfewqasdfrgt/bone-cancer-v29-train" }

# ---- kill every existing poller for this target, then start exactly one ----
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and $_.CommandLine -like "*loop_poll_v27.py*" -and
    $_.CommandLine -like "*$slug*" -and $_.CommandLine -notlike '*harness*'
}
foreach ($p in $running) {
    try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
}
if ($running) { Start-Sleep -Seconds 2 }
Remove-Item "$root\poll_sched.lock" -Force -ErrorAction SilentlyContinue

$py = (Get-Command python).Source
# Quote every argument: the project path contains spaces and Start-Process does
# not quote them for us, which silently truncates the script path.
# NO stdout redirection: Start-Process -RedirectStandardOutput makes the parent
# wait on the child's handles, which hung an interactive invocation. The poller
# writes poll_log.txt and poll_loop.out itself.
Start-Process -FilePath $py `
    -ArgumentList "`"$root\loop_poll_v27.py`"", "`"$slug`"", "0", "sched" `
    -WorkingDirectory $root `
    -WindowStyle Hidden
