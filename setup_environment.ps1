param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    & py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python ortami olusturulamadi.' }
}
& $projectPython -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Proje kutuphaneleri kurulamadi.' }
if ($CpuOnly) {
    & $projectPython -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
} else {
    & $projectPython -m pip install 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130 --disable-pip-version-check
}
if ($LASTEXITCODE -ne 0) { throw 'PyTorch kurulumu tamamlanamadi.' }
& $projectPython -m ipykernel install --prefix (Join-Path $PSScriptRoot '.venv') --name kt-project --display-name 'Knowledge Tracing (proje)'
if ($LASTEXITCODE -ne 0) { throw 'Calisma defteri cekirdegi kaydedilemedi.' }
& $projectPython scripts/check_environment.py
exit $LASTEXITCODE
