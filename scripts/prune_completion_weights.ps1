param([Parameter(Mandatory=$true)][string]$Manifest)
$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$allowedRoot = [System.IO.Path]::GetFullPath($plan.root).TrimEnd('\')
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($allowedRoot -ne (Join-Path $workspaceRoot '.cache\huggingface\hub')) { throw 'Unexpected pruning root' }
$checkedFiles = @()
foreach ($entry in $plan.files) {
    $targetPath = [System.IO.Path]::GetFullPath($entry.path)
    if (-not $targetPath.StartsWith($allowedRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Target escapes weight cache' }
    if ($targetPath -match 'models--Mapika--decider-4b|models--vllm-sr--Decision-2.0-Nox-4B') { throw 'Protected model' }
    if ([System.IO.Path]::GetExtension($targetPath) -notin @('.safetensors','.pt','.bin','.gguf')) { throw 'Only checkpoint weight files may be pruned' }
    if (-not (Test-Path -LiteralPath $targetPath)) { continue }
    $item = Get-Item -LiteralPath $targetPath -Force
    if ($item.PSIsContainer) { throw 'Expected a weight file, not a directory' }
    $ancestor = $item
    while ($null -ne $ancestor -and $ancestor.FullName -ne $workspaceRoot) {
        if ($ancestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'Refusing a reparse point' }
        $ancestor = Get-Item -LiteralPath ([System.IO.Path]::GetDirectoryName($ancestor.FullName)) -Force
    }
    if ($item.Length -ne $entry.bytes) { throw 'Weight size changed after verification' }
    if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'Weight hash changed after verification' }
    $checkedFiles += $targetPath
}
foreach ($targetPath in $checkedFiles) { Remove-Item -LiteralPath $targetPath -Force }
Write-Output ('Removed {0} verified temporary weight files.' -f $checkedFiles.Count)
