# Signed Windows release - NSIS installer (031, Part C).
#
# Runs ON the Windows box (Windows cannot cross-compile from mac). Mirrors
# `just release-mac`: build the web app, build the Tauri NSIS installer, verify
# the bundled ONNX Runtime + DirectML DLLs landed next to the exe, then sign the
# exe + installer with the Certum cloud cert IF $env:CERTUM_THUMBPRINT is set.
# With no Certum env it emits an UNSIGNED installer for testing (023 cert order
# is still pending) rather than hard-failing.
#
# Run via `just release-win`, or standalone:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\exports\release_win.ps1

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $repo

if (-not (Test-Path "apps\desktop\src-tauri\vendor\onnxruntime\x86_64-pc-windows-msvc\onnxruntime.dll")) {
    throw "ONNX Runtime not vendored - run 'just provision-ort-win' first"
}

Write-Output "=== web + desktop bundle ==="
pnpm --dir apps/web build
if ($LASTEXITCODE -ne 0) { throw "web build failed" }
pnpm --dir apps/desktop tauri build
if ($LASTEXITCODE -ne 0) { throw "tauri build failed" }

# NSIS installer is produced by bundle.targets "all".
$installer = Get-ChildItem "target\release\bundle\nsis\*-setup.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $installer) { throw "NSIS installer not found under target\release\bundle\nsis" }
$exe = Get-ChildItem "target\release\Ferment.exe" -ErrorAction SilentlyContinue | Select-Object -First 1

# Invariant 1/2: the ORT + DirectML DLLs must sit next to the exe (default DLL
# search) so a fresh box runs inference with zero extra downloads.
if ($exe) {
    $exeDir = Split-Path $exe.FullName
    foreach ($dll in @("onnxruntime.dll", "DirectML.dll", "onnxruntime_providers_shared.dll")) {
        if (Test-Path (Join-Path $exeDir $dll)) {
            Write-Output "ok: $dll beside exe"
        } else {
            Write-Warning "MISSING $dll beside exe ($exeDir) - check bundle.resources layout in tauri.windows.conf.json"
        }
    }
}

# Sign only when the Certum cloud cert is wired (023 prereq). SimplySign presents
# the OV cert to the Windows cert store; sign by its SHA-1 thumbprint.
if ($env:CERTUM_THUMBPRINT) {
    $signtool = if ($env:SIGNTOOL) { $env:SIGNTOOL } else { "signtool.exe" }
    $timestamp = if ($env:CERTUM_TIMESTAMP_URL) { $env:CERTUM_TIMESTAMP_URL } else { "http://time.certum.pl" }
    $targets = @($installer.FullName)
    if ($exe) { $targets = @($exe.FullName) + $targets }
    foreach ($f in $targets) {
        Write-Output "signing $f"
        & $signtool sign /sha1 $env:CERTUM_THUMBPRINT /fd sha256 /tr $timestamp /td sha256 $f
        if ($LASTEXITCODE -ne 0) { throw "signtool failed on $f" }
    }
    & $signtool verify /pa $installer.FullName
    Write-Output "release-win: signed $($installer.Name)"
} else {
    Write-Warning "CERTUM_THUMBPRINT unset - emitting UNSIGNED installer (023 cert order pending)."
    Write-Warning "Note: to sign the exe *inside* the installer once Certum is wired, set"
    Write-Warning "bundle.windows.signCommand in tauri.conf.json so Tauri signs before NSIS packs."
}

Write-Output "release-win: done - $($installer.FullName)"
