param([switch]$Cpu, [switch]$Quick, [int]$BatchSize = 8)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Run .\scripts\setup.ps1 first.' }
$sampleSize = if ($Quick) { 100 } else { 0 }
$validationSize = if ($Quick) { 100 } else { 0 }
$device = if ($Cpu) { 'cpu' } else { 'cuda' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$dataPath = "data/prepared/$stamp.json"
$outputPath = "results/$stamp"
& $venvPython -m laya_bench prepare --test-size $sampleSize --validation-size $validationSize --output $dataPath
if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed.' }
& $venvPython -m laya_bench run --data $dataPath --output $outputPath --device $device --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed. Inspect the partial summary; it is not a completed result.' }
Write-Host "Open $projectRoot\$outputPath\report.html for the business report."

