param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11, then rerun setup.' }
}
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$torchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
& $venvPython -m pip install 'torch==2.8.0' --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $venvPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Benchmark verification failed.' }
Write-Host 'Setup complete. Run .\scripts\benchmark.ps1 to evaluate both models.'

