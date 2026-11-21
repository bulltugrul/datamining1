$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Proje ortami bulunamadi. Once setup_environment.ps1 dosyasini calistirin.'
}
Set-Location -LiteralPath $projectRoot
& $projectPython (Join-Path $projectRoot 'scripts\launch_jupyter.py')
exit $LASTEXITCODE
