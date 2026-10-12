<#
.SYNOPSIS
    Pre-flight audit script for xau-bot deployment on Windows VPS.
.DESCRIPTION
    Validates operating system, Python 3.12+, W32Time/NTP clock synchronization,
    MetaTrader 5 installation, Common\Files read/write permissions,
    environment variables, Windows Update reboot policy, and network reachability.
.PARAMETER Detailed
    Displays verbose output for each inspection step.
.PARAMETER Json
    Outputs results in structured JSON format for automation pipelines.
.EXAMPLE
    pwsh -File scripts\verify-vps.ps1 -Detailed
#>
[CmdletBinding()]
param(
    [switch]$Detailed,
    [switch]$Json
)

$ErrorActionPreference = 'Continue'

$isWin = if (Get-Variable -Name IsWindows -ErrorAction SilentlyContinue) {
    $IsWindows
} else {
    [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
}

$results = [System.Collections.Generic.List[PSObject]]::new()

function Add-CheckResult {
    param(
        [string]$Category,
        [string]$Name,
        [string]$Status, # PASS, WARN, FAIL
        [string]$Details
    )
    $obj = [PSCustomObject]@{
        Category = $Category
        Check    = $Name
        Status   = $Status
        Details  = $Details
    }
    $results.Add($obj) | Out-Null
}

# 1. Operating System & Architecture
if (-not $isWin) {
    Add-CheckResult -Category "OS" -Name "Platform" -Status "FAIL" -Details "Non-Windows OS detected. MT5 IPC requires Windows."
} else {
    $is64 = [System.Environment]::Is64BitOperatingSystem
    if ($is64) {
        $osVer = [System.Environment]::OSVersion.VersionString
        Add-CheckResult -Category "OS" -Name "Architecture" -Status "PASS" -Details "64-bit Windows ($osVer)"
    } else {
        Add-CheckResult -Category "OS" -Name "Architecture" -Status "FAIL" -Details "32-bit Windows detected. MetaTrader 5 64-bit required."
    }
}

# 2. Python 3.12+ Environment
$pythonCmd = $null
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$botRootDir = Split-Path -Parent $scriptDir
$venvPython = Join-Path $botRootDir ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $venvPython) {
    $pythonCmd = $venvPython
} else {
    $foundPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($foundPython) {
        $pythonCmd = $foundPython.Source
    }
}

if ($null -eq $pythonCmd) {
    Add-CheckResult -Category "Runtime" -Name "Python" -Status "FAIL" -Details "Python executable not found in .venv or PATH."
} else {
    try {
        $verOut = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
        $majorMinor = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($majorMinor -eq "3.12" -or ([version]$verOut -ge [version]"3.12.0")) {
            Add-CheckResult -Category "Runtime" -Name "Python" -Status "PASS" -Details "Python $verOut located at $pythonCmd"
        } else {
            Add-CheckResult -Category "Runtime" -Name "Python" -Status "WARN" -Details "Python version is $verOut (Recommended: 3.12.x)."
        }
    } catch {
        Add-CheckResult -Category "Runtime" -Name "Python" -Status "FAIL" -Details "Failed to execute Python: $_"
    }
}

# 3. Time Synchronization & W32Time / NTP
if ($isWin) {
    try {
        $timeService = Get-Service -Name "w32time" -ErrorAction SilentlyContinue
        if ($null -eq $timeService) {
            Add-CheckResult -Category "Clock" -Name "W32Time" -Status "WARN" -Details "Windows Time (w32time) service not found."
        } elseif ($timeService.Status -eq "Running") {
            $w32tmOut = & w32tm /query /status 2>$null
            $sourceLine = ($w32tmOut | Select-String "Source:")
            $sourceStr = if ($sourceLine) { ($sourceLine -replace "Source:\s*", "").Trim() } else { "Active" }
            Add-CheckResult -Category "Clock" -Name "W32Time" -Status "PASS" -Details "Service Running. Source: $sourceStr"
        } else {
            Add-CheckResult -Category "Clock" -Name "W32Time" -Status "WARN" -Details "Service status is $($timeService.Status). Start with 'net start w32time'."
        }
    } catch {
        Add-CheckResult -Category "Clock" -Name "W32Time" -Status "WARN" -Details "Could not query time service: $_"
    }
}

# 4. MetaTrader 5 Terminal Installation
$terminalPaths = @(
    "C:\Program Files\MetaTrader 5\terminal64.exe",
    "C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    (Join-Path $env:LOCALAPPDATA "Programs\MetaTrader 5\terminal64.exe")
)
$mt5Found = $false
foreach ($tp in $terminalPaths) {
    if (Test-Path -LiteralPath $tp) {
        $mt5Found = $true
        Add-CheckResult -Category "MT5" -Name "TerminalBinary" -Status "PASS" -Details "Found terminal64.exe at $tp"
        break
    }
}
if (-not $mt5Found) {
    Add-CheckResult -Category "MT5" -Name "TerminalBinary" -Status "WARN" -Details "terminal64.exe not in default locations. Verify MT5 installation on VPS."
}

# 5. Common\Files Read/Write Permissions (Guardian Protocol)
$commonDir = $env:XAU_GUARDIAN_COMMON_DIR
if ([string]::IsNullOrWhiteSpace($commonDir)) {
    Add-CheckResult -Category "Guardian" -Name "CommonFilesDir" -Status "FAIL" -Details "XAU_GUARDIAN_COMMON_DIR is not set. Environment variable must point to MT5 Common\Files."
} elseif (-not (Test-Path -LiteralPath $commonDir)) {
    Add-CheckResult -Category "Guardian" -Name "CommonFilesDir" -Status "FAIL" -Details "Specified Common\Files directory does not exist: $commonDir"
} else {
    Add-CheckResult -Category "Guardian" -Name "CommonFilesDir" -Status "PASS" -Details "Directory exists at $commonDir"
    $testFile = Join-Path $commonDir ".vps_verify_test.tmp"
    try {
        Set-Content -LiteralPath $testFile -Value "test" -Encoding UTF8 -Force
        $readVal = Get-Content -LiteralPath $testFile -Encoding UTF8
        Remove-Item -LiteralPath $testFile -Force
        if ($readVal -eq "test") {
            Add-CheckResult -Category "Guardian" -Name "CommonFilesWrite" -Status "PASS" -Details "R/W permissions confirmed in $commonDir"
        } else {
            Add-CheckResult -Category "Guardian" -Name "CommonFilesWrite" -Status "FAIL" -Details "Content mismatch during write test."
        }
    } catch {
        Add-CheckResult -Category "Guardian" -Name "CommonFilesWrite" -Status "FAIL" -Details "Write test failed in $($commonDir): $_"
    }
}

# 6. Environment Variables and Secrets (Canonical Settings)
$envServer = $env:XAU_MT5_SERVER
$envLogin = $env:XAU_MT5_LOGIN
$envPass = $env:XAU_MT5_PASSWORD
$envTelegramToken = $env:XAU_TELEGRAM_TOKEN
$envTelegramChat = $env:XAU_TELEGRAM_CHAT_ID

if ([string]::IsNullOrWhiteSpace($envServer) -or [string]::IsNullOrWhiteSpace($envLogin)) {
    Add-CheckResult -Category "Config" -Name "MT5Credentials" -Status "WARN" -Details "XAU_MT5_SERVER or XAU_MT5_LOGIN not set in environment."
} else {
    $maskedLogin = if ($envLogin.Length -gt 2) {
        ("*" * ($envLogin.Length - 2)) + $envLogin.Substring($envLogin.Length - 2)
    } else {
        "**"
    }
    Add-CheckResult -Category "Config" -Name "MT5Credentials" -Status "PASS" -Details "MT5 credentials configured (Login: $maskedLogin, Server: $envServer)"
}

if ([string]::IsNullOrWhiteSpace($envPass)) {
    Add-CheckResult -Category "Config" -Name "MT5Password" -Status "WARN" -Details "XAU_MT5_PASSWORD not configured in environment."
} else {
    Add-CheckResult -Category "Config" -Name "MT5Password" -Status "PASS" -Details "MT5 password is set (masked: ********)"
}

if (-not [string]::IsNullOrWhiteSpace($envTelegramToken) -and -not [string]::IsNullOrWhiteSpace($envTelegramChat)) {
    Add-CheckResult -Category "Config" -Name "TelegramAlerts" -Status "PASS" -Details "Telegram Token and Chat ID configured (masked: ********)"
} elseif (-not [string]::IsNullOrWhiteSpace($envTelegramToken) -or -not [string]::IsNullOrWhiteSpace($envTelegramChat)) {
    Add-CheckResult -Category "Config" -Name "TelegramAlerts" -Status "WARN" -Details "Incomplete Telegram config: both XAU_TELEGRAM_TOKEN and XAU_TELEGRAM_CHAT_ID are required."
} else {
    Add-CheckResult -Category "Config" -Name "TelegramAlerts" -Status "WARN" -Details "Telegram alerts not configured (XAU_TELEGRAM_TOKEN and XAU_TELEGRAM_CHAT_ID unset)."
}

# 7. Windows Update Automatic Reboot Policy
if ($isWin) {
    $regPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU"
    if (Test-Path -LiteralPath $regPath) {
        $noAutoReboot = (Get-ItemProperty -LiteralPath $regPath -Name "NoAutoRebootWithLoggedOnUsers" -ErrorAction SilentlyContinue).NoAutoRebootWithLoggedOnUsers
        if ($noAutoReboot -eq 1) {
            Add-CheckResult -Category "WindowsUpdate" -Name "RebootPolicy" -Status "PASS" -Details "NoAutoRebootWithLoggedOnUsers is ACTIVE (reboot prevented while logged on)."
        } else {
            Add-CheckResult -Category "WindowsUpdate" -Name "RebootPolicy" -Status "WARN" -Details "NoAutoRebootWithLoggedOnUsers is 0 or unset. Unscheduled reboots may occur."
        }
    } else {
        Add-CheckResult -Category "WindowsUpdate" -Name "RebootPolicy" -Status "WARN" -Details "Windows Update AU policy key not configured. Consult RUNBOOK.md for GPO steps."
    }
}

# 8. Network Connectivity
$dnsTargets = @("pool.ntp.org", "api.telegram.org")
foreach ($target in $dnsTargets) {
    try {
        $ip = [System.Net.Dns]::GetHostAddresses($target)
        if ($ip -and $ip.Length -gt 0) {
            Add-CheckResult -Category "Network" -Name "DNS_$target" -Status "PASS" -Details "Resolved to $($ip[0].IPAddressToString)"
        } else {
            Add-CheckResult -Category "Network" -Name "DNS_$target" -Status "WARN" -Details "DNS resolution returned empty."
        }
    } catch {
        Add-CheckResult -Category "Network" -Name "DNS_$target" -Status "WARN" -Details "DNS resolution failed: $_"
    }
}

# Output Generation
$failCount = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
$warnCount = @($results | Where-Object { $_.Status -eq "WARN" }).Count
$passCount = @($results | Where-Object { $_.Status -eq "PASS" }).Count

if ($Json) {
    $outObj = [PSCustomObject]@{
        Timestamp = (Get-Date).ToString("o")
        Summary   = @{
            Pass = $passCount
            Warn = $warnCount
            Fail = $failCount
        }
        Checks    = $results
    }
    $outObj | ConvertTo-Json -Depth 4
} else {
    Write-Host "`n=== XAU-BOT VPS PRE-FLIGHT VERIFICATION REPORT ===" -ForegroundColor Cyan
    Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"

    foreach ($res in $results) {
        $color = switch ($res.Status) {
            "PASS" { "Green" }
            "WARN" { "Yellow" }
            "FAIL" { "Red" }
            Default { "White" }
        }
        $tag = "[{0}]" -f $res.Status.PadRight(4)
        Write-Host "$tag " -NoNewline -ForegroundColor $color
        Write-Host "$($res.Category.PadRight(14)) - $($res.Check.PadRight(20)) : $($res.Details)"
    }

    Write-Host "`nSummary: $passCount PASS | $warnCount WARN | $failCount FAIL" -ForegroundColor $(if ($failCount -gt 0) { "Red" } elseif ($warnCount -gt 0) { "Yellow" } else { "Green" })

    if ($failCount -gt 0) {
        Write-Host "CRITICAL: Found $failCount failing checks. Fix them before starting live trading!`n" -ForegroundColor Red
    } else {
        Write-Host "SUCCESS: Pre-flight checks passed. VPS ready for operation.`n" -ForegroundColor Green
    }
}

if ($failCount -gt 0) {
    exit 1
} else {
    exit 0
}
