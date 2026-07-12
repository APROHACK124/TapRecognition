# Convert TapRecognition ONNX model to MindSpore Lite .ms for HarmonyOS
#
# Prerequisites:
# 1. Export ONNX first:
#      python tools/export_for_harmony.py
# 2. MindSpore Lite Windows package extracted, e.g. D:\mindspore-lite-2.9.0-win-x64
# 3. MinGW-w64 runtime DLLs (libgcc_s_seh-1.dll, libstdc++-6.dll, libwinpthread-1.dll, libssp-0.dll)
#    Install MSYS2, then run in MSYS2 terminal:
#      pacman -S --noconfirm mingw-w64-x86_64-gcc

param(
    [string]$OnnxFile = "checkpoints/tap_recognition.onnx",
    [string]$OutputFile = "checkpoints/tap_recognition",
    [string]$MsLiteRoot = "D:\mindspore-lite-2.9.0-win-x64",
    [string]$MingwBin = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$requiredDlls = @(
    "libgcc_s_seh-1.dll",
    "libstdc++-6.dll",
    "libwinpthread-1.dll",
    "libssp-0.dll"
)

function Find-MingwBin {
    param([string]$CustomPath)

    if ($CustomPath -and (Test-Path $CustomPath)) {
        return $CustomPath
    }

    $candidates = @(
        "C:\msys64\mingw64\bin",
        "D:\msys64\mingw64\bin",
        "C:\mingw64\bin"
    )

    foreach ($candidate in $candidates) {
        $gcc = Join-Path $candidate "libgcc_s_seh-1.dll"
        if (Test-Path $gcc) {
            return $candidate
        }
    }

    return $null
}

function Ensure-MingwDlls {
    param(
        [string]$SourceBin,
        [string]$TargetLib
    )

    foreach ($dll in $requiredDlls) {
        $src = Join-Path $SourceBin $dll
        if (-not (Test-Path $src)) {
            throw "Missing MinGW runtime DLL: $src"
        }
        Copy-Item $src (Join-Path $TargetLib $dll) -Force
    }
}

if (-not (Test-Path $OnnxFile)) {
    Write-Host "ONNX file not found: $OnnxFile"
    Write-Host "Run: python tools/export_for_harmony.py"
    exit 1
}

$converter = Join-Path $MsLiteRoot "tools\converter\converter\converter_lite.exe"
$libDir = Join-Path $MsLiteRoot "tools\converter\lib"

if (-not (Test-Path $converter)) {
    Write-Host "MindSpore Lite converter not found: $converter"
    Write-Host "Download from: https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html"
    exit 1
}

$mingwBin = Find-MingwBin -CustomPath $MingwBin
if (-not $mingwBin) {
    Write-Host "MinGW-w64 runtime not found."
    Write-Host ""
    Write-Host "converter_lite.exe requires these DLLs:"
    foreach ($dll in $requiredDlls) { Write-Host "  - $dll" }
    Write-Host ""
    Write-Host "Install MSYS2 from https://www.msys2.org/, open MSYS2 MINGW64, then run:"
    Write-Host "  pacman -S --noconfirm mingw-w64-x86_64-gcc"
    Write-Host ""
    Write-Host "Then rerun:"
    Write-Host "  .\tools\convert_to_ms.ps1 -MingwBin C:\msys64\mingw64\bin"
    exit 1
}

Write-Host "Copying MinGW runtime from $mingwBin ..."
Ensure-MingwDlls -SourceBin $mingwBin -TargetLib $libDir

$env:PATH = "$libDir;$env:PATH"
$env:GLOG_v = "1"

Write-Host "Converting ONNX -> MS..."
& $converter --fmk=ONNX --modelFile=$OnnxFile --outputFile=$OutputFile --inputShape="imu:1,64,6"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Conversion failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Success: ${OutputFile}.ms"
