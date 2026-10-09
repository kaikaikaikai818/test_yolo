param(
    [string]$Prompt = "screwdriver"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Entry = Join-Path $ProjectRoot "grasp_test.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "项目虚拟环境不存在：$Python"
}
if (-not (Test-Path -LiteralPath $Entry)) {
    throw "测试入口不存在：$Entry"
}

Push-Location $ProjectRoot
try {
    & $Python $Entry --prompt $Prompt --stage vision --check-calib
    if ($LASTEXITCODE -ne 0) {
        throw "视觉测试退出，错误码：$LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
