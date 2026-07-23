param(
  [string]$DatasetDir = "reports\material_sorting_labelimg_yellow_dataset_20260722"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LabelImg = Join-Path $Root ".venv39\Scripts\labelImg.exe"
$Dataset = Join-Path $Root $DatasetDir
$Workspace = Join-Path $Dataset "labelimg_workspace"
$Classes = Join-Path $Dataset "classes.txt"
$SaveDir = Join-Path $Dataset "label"

if (!(Test-Path $LabelImg)) {
  throw "LabelImg executable not found: $LabelImg"
}
if (!(Test-Path $Workspace)) {
  throw "LabelImg image workspace not found: $Workspace"
}
if (!(Test-Path $Classes)) {
  throw "classes.txt not found: $Classes"
}
if (!(Test-Path $SaveDir)) {
  throw "label save directory not found: $SaveDir"
}

$SaveClasses = Join-Path $SaveDir "classes.txt"
$WorkspaceClasses = Join-Path $Workspace "classes.txt"
Copy-Item -LiteralPath $Classes -Destination $SaveClasses -Force
Copy-Item -LiteralPath $Classes -Destination $WorkspaceClasses -Force

Write-Host "[labelImg] image_dir=$Workspace"
Write-Host "[labelImg] classes=$Classes"
Write-Host "[labelImg] save_dir=$SaveDir"
Write-Host "[labelImg] synced classes.txt to $SaveClasses"
Write-Host "[labelImg] In the app, make sure the left toolbar shows YOLO. If it shows PascalVOC, click it once to switch to YOLO."

& $LabelImg $Workspace $Classes $SaveDir
