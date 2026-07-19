# Logon-triggered auto-recovery: relaunch Pass C after a reboot.
# Registered as scheduled task AAAI_PassC_AutoResume (see decisions.md D17).
# Safe to fire any time: passC.ps1 exits if workers already run, skips
# finished runs, and resumes unfinished ones from resume.pt.
$repo = Split-Path $PSScriptRoot -Parent
Start-Sleep -Seconds 90  # let the system settle after logon
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\passC.ps1" *>> "$repo\logs\passC_auto.log"
