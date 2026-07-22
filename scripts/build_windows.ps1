param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$BuildPath = Join-Path $RepoRoot "build\terraforge"
$ExecutablePath = Join-Path $RepoRoot "dist\TerraForge.exe"

if ($Clean) {
    $ResolvedRoot = [System.IO.Path]::GetFullPath($RepoRoot)
    $ResolvedBuild = [System.IO.Path]::GetFullPath($BuildPath)
    if (-not $ResolvedBuild.StartsWith($ResolvedRoot + [System.IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing to clean a build path outside the repository"
    }
    Remove-Item -LiteralPath $BuildPath -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ExecutablePath -Force -ErrorAction SilentlyContinue
}

python -m pip install -e ".[build]"
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE"
}
python -m PyInstaller --noconfirm --clean "packaging\terraforge.spec"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
    throw "PyInstaller completed without creating dist\TerraForge.exe"
}
Write-Host "Built dist\TerraForge.exe" -ForegroundColor Green
