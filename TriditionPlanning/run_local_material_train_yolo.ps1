param(
  [string]$DataYaml = "reports\material_sorting_yolo_dataset_yellow_demo\data.yaml",
  [string]$Model = "competition_workspace\material_sorting\perception\checkpoints\material_box.pt",
  [int]$Epochs = 1,
  [int]$ImageSize = 320,
  [int]$Batch = 4,
  [string]$Device = "cpu",
  [string]$Name = "yellow_demo",
  [string]$ExportCheckpoint = "competition_workspace\material_sorting\perception\checkpoints\material_box_yellow_demo.pt"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv39\Scripts\python.exe"
$Train = Join-Path $Root "competition_workspace\material_sorting\perception\train_yolo_material.py"

if (!(Test-Path $Python)) {
  throw "Python venv not found: $Python"
}
if (!(Test-Path $Train)) {
  throw "Train script not found: $Train"
}

$env:YOLO_CONFIG_DIR = Join-Path $Root ".ultralytics"

Write-Host "[train-yolo] data=$DataYaml model=$Model epochs=$Epochs imgsz=$ImageSize device=$Device"
& $Python $Train `
  --data-yaml $DataYaml `
  --model $Model `
  --epochs $Epochs `
  --imgsz $ImageSize `
  --batch $Batch `
  --device $Device `
  --name $Name `
  --export-checkpoint $ExportCheckpoint
