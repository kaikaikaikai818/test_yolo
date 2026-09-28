param([string]$Source = '')
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if ($Source) {
    & $pythonPath (Join-Path $PSScriptRoot 'predict.py') --source $Source
} else {
    & $pythonPath (Join-Path $PSScriptRoot 'predict.py')
}
exit $LASTEXITCODE

