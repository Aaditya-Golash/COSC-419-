param(
  [string]$DataRoot = "SoccerNet/jersey-2023",
  [string]$TinyOut = "data/tiny5",
  [string]$SanityConfig = "configs/sanity.yaml",
  [switch]$SkipSubset,
  [switch]$SkipSanity
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

function Resolve-Python {
  $candidates = @(
    (Join-Path $repoRoot ".venv_g9\Scripts\python.exe"),
    (Join-Path $repoRoot ".venv\Scripts\python.exe")
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }
  return "python"
}

function Resolve-JerseyRoot([string]$candidateRoot) {
  $resolved = Resolve-Path -LiteralPath $candidateRoot -ErrorAction SilentlyContinue
  if (-not $resolved) {
    throw "Dataset root does not exist: $candidateRoot"
  }
  $root = [string]$resolved

  $canonicalImages = Join-Path $root "train\images"
  $canonicalGt = Join-Path $root "train\train_gt.json"
  if ((Test-Path $canonicalImages) -and (Test-Path $canonicalGt)) {
    return $root
  }

  # Handles partial extraction layouts like SoccerNet/jersey-2023/train/train/images
  $nestedImages = Join-Path $root "train\train\images"
  $nestedGt = Join-Path $root "train\train\train_gt.json"
  if ((Test-Path $nestedImages) -and (Test-Path $nestedGt)) {
    return (Join-Path $root "train")
  }

  throw "Could not find expected train/images + train_gt.json under: $root"
}

$pythonExe = Resolve-Python
$jerseyRoot = Resolve-JerseyRoot $DataRoot

$logDir = Join-Path $repoRoot "meeting_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "meeting_demo_$timestamp.log"

Write-Host "Repo root: $repoRoot"
Write-Host "Python: $pythonExe"
Write-Host "Dataset root used for scripts: $jerseyRoot"
Write-Host "Log file: $logPath"

if (-not $SkipSubset) {
  Write-Host "`n[1/2] Creating tiny 5-clip subset..."
  & $pythonExe "scripts/make_tiny_subset.py" --data_root $jerseyRoot --split train --num_clips 5 --out_dir $TinyOut 2>&1 | Tee-Object -FilePath $logPath -Append
  if ($LASTEXITCODE -ne 0) {
    throw "Subset creation failed."
  }
} else {
  Write-Host "`n[1/2] Skipped subset creation (--SkipSubset)."
}

if (-not $SkipSanity) {
  Write-Host "`n[2/2] Running sanity train loop..."
  & $pythonExe "src/train_sanity.py" --config $SanityConfig 2>&1 | Tee-Object -FilePath $logPath -Append
  if ($LASTEXITCODE -ne 0) {
    throw "Sanity run failed."
  }
} else {
  Write-Host "`n[2/2] Skipped sanity run (--SkipSanity)."
}

Write-Host "`nMeeting demo flow completed."
