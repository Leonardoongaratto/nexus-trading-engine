<#
.SYNOPSIS
    Windows Service management script for xau-bot using NSSM (Non-Sucking Service Manager).
.DESCRIPTION
    Provides automated commands to Install, Start, Stop, Restart, Query Status,
    View Logs, and Remove the xau-bot background Windows Service.
.PARAMETER Action
    The action to execute: Install, Start, Stop, Restart, Status, Remove, Logs.
.PARAMETER ServiceName
    The name of the Windows service (Default: XauBot).
.PARAMETER Port
    The health and metrics HTTP port (Default: 8000).
.PARAMETER Broker
    The execution broker adapter: mt5 or sim (Default: mt5).
.PARAMETER NssmPath
    Custom path to nssm.exe if not in system PATH.
.PARAMETER DryRun
    Outputs the planned commands without executing them (ideal for verification).
.EXAMPLE
    pwsh -File scripts\service-nssm.ps1 -Action Install -Broker mt5 -Port 8000
.EXAMPLE
    pwsh -File scripts\service-nssm.ps1 -Action Status
.EXAMPLE
    pwsh -File scripts\service-nssm.ps1 -Action Install -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("Install", "Start", "Stop", "Restart", "Status", "Remove", "Logs")]
    [string]$Action,

    [string]$ServiceName = "XauBot",
    [int]$Port = 8000,
    [ValidateSet("mt5", "sim")]
    [string]$Broker = "mt5",
    [string]$NssmPath = "",
    [switch]$DryRun,
    [Parameter(DontShow)]
    [switch]$SkipAdminCheck
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$botRootDir = (Resolve-Path (Join-Path $scriptDir "..")).Path
$venvPython = Join-Path $botRootDir ".venv\Scripts\python.exe"

function Find-Nssm {
    param([string]$customPath)
    if (-not [string]::IsNullOrWhiteSpace($customPath)) {
        if (Test-Path -LiteralPath $customPath) {
            return (Resolve-Path $customPath).Path
        }
        throw "Specified NSSM executable not found at: $customPath"
    }

    $cmd = Get-Command "nssm.exe" -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        (Join-Path $scriptDir "nssm.exe"),
        "C:\tools\nssm\win64\nssm.exe",
        "C:\tools\nssm\nssm.exe",
        "C:\nssm\win64\nssm.exe",
        "C:\nssm\nssm.exe"
    )

    foreach ($cand in $candidates) {
        if (Test-Path -LiteralPath $cand) {
            return $cand
        }
    }

    return $null
}

$nssmActions = @("Install", "Start", "Stop", "Restart", "Remove")
$nssmExe = if ($Action -in $nssmActions) { Find-Nssm -customPath $NssmPath } else { $null }

function Assert-Administrator {
    if ($SkipAdminCheck) {
        return
    }
    $isWin = if (Get-Variable -Name IsWindows -ErrorAction SilentlyContinue) {
        $IsWindows
    } else {
        [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
    }
    if ($isWin) {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = [Security.Principal.WindowsPrincipal]::new($identity)
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw "Administrative privileges required. Please run PowerShell as Administrator."
        }
    }
}

function Run-CommandOrDryRun {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    $cmdStr = "$Executable " + ($Arguments -join " ")
    if ($DryRun) {
        Write-Host "[DRY-RUN] $cmdStr" -ForegroundColor Cyan
        return
    }
    Write-Host "[EXEC] $cmdStr" -ForegroundColor DarkGray
    & $Executable @Arguments | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $code = $LASTEXITCODE
        [Console]::Error.WriteLine("Command failed with exit code ${code}: $cmdStr")
        exit $code
    }
}

switch ($Action) {
    "Install" {
        if (-not $DryRun) { Assert-Administrator }
        Write-Host "`n=== Installing xau-bot Windows Service ($ServiceName) ===" -ForegroundColor Green

        if (-not (Test-Path -LiteralPath $venvPython)) {
            $pyCmd = Get-Command "python.exe" -ErrorAction SilentlyContinue
            if ($pyCmd) {
                Write-Warning "Local virtualenv (.venv) not found. Falling back to system Python: $($pyCmd.Source). Recommended: run 'python -m venv .venv && .\.venv\Scripts\pip install -e .' first."
                $venvPython = $pyCmd.Source
            } else {
                throw "Python executable not found in .venv or PATH. Run 'python -m venv .venv' first."
            }
        }

        if ($null -eq $nssmExe -and -not $DryRun) {
            throw "nssm.exe not found. Install NSSM or specify -NssmPath. (See RUNBOOK.md)"
        }
        if ($null -eq $nssmExe) {
            $nssmExe = "nssm.exe"
        }

        $logsDir = Join-Path $botRootDir "logs"
        if (-not (Test-Path -LiteralPath $logsDir) -and -not $DryRun) {
            New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
        }

        $stdoutLog = Join-Path $logsDir "service_stdout.log"
        $stderrLog = Join-Path $logsDir "service_stderr.log"
        $appArgs = "-m xau_bot.cli run --broker $Broker --port $Port --log-json"

        Write-Host "Service Name : $ServiceName"
        Write-Host "Bot Directory: $botRootDir"
        Write-Host "Python Path  : $venvPython"
        Write-Host "Arguments    : $appArgs"
        Write-Host "Logs Path    : $logsDir`n"

        Run-CommandOrDryRun $nssmExe @("install", $ServiceName, $venvPython, $appArgs)
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppDirectory", $botRootDir)
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "DisplayName", "XAU/USD SMC Trading Robot ($Broker)")
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "Description", "Automated SMC Trading Bot for MT5 with Guardian Heartbeat and Kill Switches.")
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "Start", "SERVICE_AUTO_START")

        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppStdout", $stdoutLog)
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppStderr", $stderrLog)
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppRotateFiles", "1")
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppRotateOnline", "1")
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppRotateSeconds", "86400")
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppRotateBytes", "10485760")

        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppRestartDelay", "5000")
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppThrottle", "1500")
        Run-CommandOrDryRun $nssmExe @("set", $ServiceName, "AppExit", "Default", "Restart")

        Write-Host "`nService $ServiceName successfully configured." -ForegroundColor Green
        Write-Host "Start service with: pwsh scripts\service-nssm.ps1 -Action Start"
    }

    "Start" {
        if (-not $DryRun) { Assert-Administrator }
        Write-Host "Starting Windows Service $ServiceName..." -ForegroundColor Cyan
        if ($null -eq $nssmExe -and -not $DryRun) { throw "nssm.exe not found." }
        if ($null -eq $nssmExe) { $nssmExe = "nssm.exe" }
        Run-CommandOrDryRun $nssmExe @("start", $ServiceName)
    }

    "Stop" {
        if (-not $DryRun) { Assert-Administrator }
        Write-Host "Stopping Windows Service $ServiceName..." -ForegroundColor Yellow
        if ($null -eq $nssmExe -and -not $DryRun) { throw "nssm.exe not found." }
        if ($null -eq $nssmExe) { $nssmExe = "nssm.exe" }
        Run-CommandOrDryRun $nssmExe @("stop", $ServiceName)
    }

    "Restart" {
        if (-not $DryRun) { Assert-Administrator }
        Write-Host "Restarting Windows Service $ServiceName..." -ForegroundColor Cyan
        if ($null -eq $nssmExe -and -not $DryRun) { throw "nssm.exe not found." }
        if ($null -eq $nssmExe) { $nssmExe = "nssm.exe" }
        Run-CommandOrDryRun $nssmExe @("restart", $ServiceName)
    }

    "Status" {
        Write-Host "Querying status for $ServiceName..." -ForegroundColor Cyan
        $winService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        if ($winService) {
            Write-Host "Windows Service: $($winService.DisplayName)"
            Write-Host "Status         : $($winService.Status)" -ForegroundColor $(if ($winService.Status -eq "Running") { "Green" } else { "Yellow" })
            Write-Host "StartType      : $($winService.StartType)"
        } else {
            Write-Host "Service $ServiceName is not installed." -ForegroundColor Yellow
        }

        # Query health endpoint
        $healthUrl = "http://127.0.0.1:$Port/health"
        Write-Host "Checking health probe at $healthUrl..."
        try {
            $resp = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3 -ErrorAction Stop
            Write-Host "Health Probe Response:" -ForegroundColor Green
            $resp | ConvertTo-Json -Depth 3 | Write-Host
        } catch {
            Write-Host "Health probe could not connect ($($_.Exception.Message))" -ForegroundColor DarkGray
        }
    }

    "Remove" {
        if (-not $DryRun) { Assert-Administrator }
        Write-Host "Removing Windows Service $ServiceName..." -ForegroundColor Red
        if ($null -eq $nssmExe -and -not $DryRun) { throw "nssm.exe not found." }
        if ($null -eq $nssmExe) { $nssmExe = "nssm.exe" }
        Run-CommandOrDryRun $nssmExe @("remove", $ServiceName, "confirm")
        Write-Host "Service $ServiceName removed." -ForegroundColor Green
    }

    "Logs" {
        $logsDir = Join-Path $botRootDir "logs"
        $stdoutLog = Join-Path $logsDir "service_stdout.log"
        $stderrLog = Join-Path $logsDir "service_stderr.log"

        Write-Host "=== Inspecting Service Logs in $logsDir ===" -ForegroundColor Cyan
        if (Test-Path -LiteralPath $stderrLog) {
            Write-Host "`n--- Last 20 lines of stderr ($stderrLog) ---" -ForegroundColor Yellow
            Get-Content -LiteralPath $stderrLog -Tail 20 -Encoding UTF8
        }
        if (Test-Path -LiteralPath $stdoutLog) {
            Write-Host "`n--- Last 30 lines of stdout ($stdoutLog) ---" -ForegroundColor Green
            Get-Content -LiteralPath $stdoutLog -Tail 30 -Encoding UTF8
        }
        if (-not (Test-Path -LiteralPath $stdoutLog) -and -not (Test-Path -LiteralPath $stderrLog)) {
            Write-Host "No log files found in $logsDir." -ForegroundColor Yellow
        }
    }
}
