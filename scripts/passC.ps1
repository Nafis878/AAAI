# Pass C (overnight): extension pass then 5th core seed, sequentially (D16).
# Idempotent + reboot-safe: --skip-existing skips finished runs; unfinished
# runs resume exactly from their latest resume.pt (train.py --resume-every).
# Guard: refuses to double-launch if grid workers are already running.
Set-Location $PSScriptRoot\..

$already = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'src\.(train|launch)' }
if ($already) {
    Write-Output "passC: grid already running ($($already.Count) processes) - exiting"
    exit 0
}

& .\.venv\Scripts\python.exe -m src.launch --preset extend40 --skip-existing
& .\.venv\Scripts\python.exe -m src.launch --preset core-ext --skip-existing
Write-Output "PASS C COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
