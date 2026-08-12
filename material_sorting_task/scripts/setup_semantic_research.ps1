[CmdletBinding()]
param(
  [switch]$ML,
  [switch]$SLM,
  [switch]$Runtime,
  [switch]$All
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$modelDir = Join-Path $root 'semantic_research\artifacts\slm'
$model = Join-Path $modelDir 'qwen2.5-3b-instruct-q4_k_m.gguf'
$url = 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/cc1e68eea5f05f88f41a6de1fc73110178f23715/qwen2.5-3b-instruct-q4_k_m.gguf?download=true'
$expectedSha256 = '626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d'

if ($All) { $ML = $true; $SLM = $true; $Runtime = $true }
if (-not ($ML -or $SLM -or $Runtime)) {
  Write-Host '用法: .\scripts\setup_semantic_research.ps1 [-ML] [-SLM] [-Runtime] [-All]'
  Write-Host '  -ML       从 train split 重训 ml_slots_v2.joblib'
  Write-Host '  -SLM      下载并校验可选的 2.1GB Qwen GGUF'
  Write-Host '  -Runtime  安装 research-only Python 依赖'
  exit 0
}

Push-Location $root
try {
  if ($Runtime) {
    python -m pip install -r semantic_research\requirements-research.txt
  }
  if ($ML) {
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONPATH = $root
    python -m semantic_research.train_ml `
      --dataset semantic_research\data\text_eval.jsonl `
      --splits train --seed 7 `
      --out semantic_research\artifacts\ml_slots_v2.joblib
  }
  if ($SLM) {
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    if (-not (Test-Path -LiteralPath $model)) {
      $part = "$model.part"
      try {
        Start-BitsTransfer -Source $url -Destination $part -DisplayName 'Qwen2.5-3B GGUF'
      } catch {
        Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing
      }
      Move-Item -LiteralPath $part -Destination $model -Force
    }
    $actual = (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expectedSha256) {
      throw "SHA256 mismatch: expected=$expectedSha256 actual=$actual"
    }
    Write-Host "verified $model"
  }
} finally {
  Pop-Location
}
Write-Host 'semantic research setup complete; formal control path was not modified'
