# Launch the 15-minute poller as a detached process.
# Invoked by the Windows Scheduled Task every 15 minutes; the loop's own lock file
# prevents duplicate instances, so a server restart is repaired within one cycle.

$root = "C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector"
$slugFile = "$root\poll_target.txt"
if (Test-Path $slugFile) {
    $slug = (Get-Content $slugFile -Raw).Trim()
} else {
    $slug = "juyhtrfewqasdfrgt/bone-cancer-v28-train"
}
if (-not $slug) { $slug = "juyhtrfewqasdfrgt/bone-cancer-v28-train" }

if (Test-Path "$root\poll_loop.lock") {
    $pidInLock = (Get-Content "$root\poll_loop.lock" -Raw).Trim()
    if ($pidInLock -and (Get-Process -Id $pidInLock -ErrorAction SilentlyContinue)) {
        exit 0
    }
}

$py = (Get-Command python).Source
# Quote every argument: the project path contains spaces and Start-Process does
# not quote them for us, which silently truncates the script path.
Start-Process -FilePath $py `
    -ArgumentList "`"$root\loop_poll_v27.py`"", "`"$slug`"" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$root\poll_loop.out" `
    -RedirectStandardError "$root\poll_loop.err"
