param(
  [ValidateSet("color", "yolo", "yolo_sam")]
  [string]$Backend = "yolo_sam",
  [switch]$ShowWindow
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv39\Scripts\python.exe"
$Demo = Join-Path $Root "examples\tasks_mmk2\material_sorting_vision_demo.py"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (!(Test-Path $Python)) {
  throw "Python venv not found: $Python"
}
if (!(Test-Path $Demo)) {
  throw "Vision demo not found: $Demo"
}

$CommonArgs = @()
if (!$ShowWindow) {
  $CommonArgs += "--headless"
  $CommonArgs += "--no-sync"
}
$CommonArgs += @("--backend", $Backend, "--warmup-steps", "2")

Write-Host "[local-vision] backend=$Backend"
Write-Host "[local-vision] camera=2 head_cam, table objects"
& $Python $Demo @CommonArgs --camera 2 --output-dir "reports\material_sorting_local_${Backend}_head_$Stamp"

Write-Host "[local-vision] camera=1 material_shelf_debug_cam, shelf objects"
& $Python $Demo @CommonArgs --camera 1 --output-dir "reports\material_sorting_local_${Backend}_shelf_$Stamp"

Write-Host "[local-vision] done"
