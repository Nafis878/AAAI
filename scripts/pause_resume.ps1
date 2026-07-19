# Suspend/resume all grid training processes (launcher + workers) in place.
# Usage:  .\scripts\pause_resume.ps1 -Action pause
#         .\scripts\pause_resume.ps1 -Action resume
param([Parameter(Mandatory = $true)][ValidateSet("pause", "resume")][string]$Action)

$sig = @'
using System;
using System.Runtime.InteropServices;
public static class ProcCtl {
    [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
    [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
    [DllImport("kernel32.dll")] public static extern IntPtr OpenProcess(uint a, bool i, int pid);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
'@
if (-not ([System.Management.Automation.PSTypeName]'ProcCtl').Type) { Add-Type -TypeDefinition $sig }

$procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'src\.(train|launch)' }

foreach ($p in $procs) {
    $h = [ProcCtl]::OpenProcess(0x1F0FFF, $false, $p.ProcessId)
    if ($h -eq [IntPtr]::Zero) { Write-Output "SKIP pid $($p.ProcessId) (no handle)"; continue }
    if ($Action -eq "pause") { [void][ProcCtl]::NtSuspendProcess($h) } else { [void][ProcCtl]::NtResumeProcess($h) }
    [void][ProcCtl]::CloseHandle($h)
    $tag = if ($p.CommandLine -match 'src\.launch') { "launcher" } else { "worker" }
    Write-Output "$Action pid $($p.ProcessId) ($tag)"
}
Write-Output "$Action complete: $($procs.Count) processes"
