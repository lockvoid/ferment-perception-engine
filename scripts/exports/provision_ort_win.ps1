# Provision the Windows ML runtime for development and tests.
#
# Fetches the self-contained Windows ML NuGet (Microsoft.Windows.AI.MachineLearning)
# pinned by URL and SHA-256, and copies its x64 runtime into this repo's vendor
# path, where tests (ORT_DYLIB_PATH) and the consuming app take it from:
#   onnxruntime.dll                          ONNX Runtime 1.28.3, DirectML built in
#   DirectML.dll                             DirectML 1.15.4
#   Microsoft.Windows.AI.MachineLearning.dll the execution-provider catalog
# HyprZero's scripts/windows_inputs.py ships the same three files from the same
# package; change both pins together. The package's license permits
# redistributing the files it binplaces with an application.
#
# User-space only (no admin, no Python). Run on the Windows box via:
#   just provision-ort-win
#   # or standalone:
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\exports\provision_ort_win.ps1

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$version = "2.5.96"
$sha256 = "35d6e84feb4142e8cecaf36704fa3ab0fc8251d892850fb80e8341f7d09653a3"
$url = "https://api.nuget.org/v3-flatcontainer/microsoft.windows.ai.machinelearning/$version/" +
    "microsoft.windows.ai.machinelearning.$version.nupkg"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$dest = Join-Path $repo "vendor\onnxruntime\x86_64-pc-windows-msvc"
$package = Join-Path $env:TEMP "ferment-windows-ml-$version.nupkg"

Write-Output "=== Windows ML $version ==="
if (-not ((Test-Path $package) -and (Get-FileHash $package -Algorithm SHA256).Hash -eq $sha256)) {
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $package
}
$got = (Get-FileHash $package -Algorithm SHA256).Hash.ToLower()
if ($got -ne $sha256) {
    throw "digest mismatch for $url`n  pinned  $sha256`n  fetched $got"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force -Path $dest | Out-Null
# A DirectML build of plain ONNX Runtime needed this bridge; Windows ML does
# not, and a stale copy would only mislead.
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $dest "onnxruntime_providers_shared.dll")
$archive = [System.IO.Compression.ZipFile]::OpenRead($package)
try {
    foreach ($dll in "onnxruntime.dll", "DirectML.dll", "Microsoft.Windows.AI.MachineLearning.dll") {
        $entry = $archive.GetEntry("runtimes/win-x64/native/$dll")
        if (-not $entry) {
            throw "missing runtimes/win-x64/native/$dll in $package"
        }
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $dest $dll), $true)
        Write-Output ("copied $dll ({0:N1} MB)" -f ($entry.Length / 1MB))
    }
} finally {
    $archive.Dispose()
}

Write-Output "=== PROVISION-ORT-WIN-DONE: $dest ==="
