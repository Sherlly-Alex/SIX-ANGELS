param(
  [string]$Classes = "yellow",
  [int]$Samples = 40,
  [string]$OutputDir = "",
  [int]$Seed = 7
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv39\Scripts\python.exe"
$Demo = Join-Path $Root "examples\tasks_mmk2\material_sorting_make_yolo_dataset.py"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (!(Test-Path $Python)) {
  throw "Python venv not found: $Python"
}
if (!(Test-Path $Demo)) {
  throw "Dataset script not found: $Demo"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $SafeClasses = $Classes.Replace(",", "_")
  $OutputDir = "reports\material_sorting_yolo_dataset_${SafeClasses}_$Stamp"
}

Write-Host "[make-yolo-dataset] classes=$Classes samples=$Samples output=$OutputDir"
& $Python $Demo --classes $Classes --samples $Samples --output-dir $OutputDir --seed $Seed
