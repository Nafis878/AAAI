# Pass C (overnight): extension pass then 5th core seed, sequentially (D16).
Set-Location $PSScriptRoot\..
& .\.venv\Scripts\python.exe -m src.launch --preset extend40 --skip-existing
& .\.venv\Scripts\python.exe -m src.launch --preset core-ext --skip-existing
Write-Output "PASS C COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
