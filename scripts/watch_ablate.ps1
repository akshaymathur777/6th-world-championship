# Restart ablate until data/winner.json exists. Safe because embed/eval rows resume.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"
$env:ANONYMIZED_TELEMETRY = "False"
while (-not (Test-Path "data\winner.json")) {
  Write-Host "$(Get-Date -Format o) starting ablate"
  & .\.venv\Scripts\python.exe -u -m src ablate *>> data\ablate.log
  $code = $LASTEXITCODE
  Write-Host "$(Get-Date -Format o) ablate exited $code"
  if (Test-Path "data\winner.json") { break }
  Start-Sleep -Seconds 5
}
Write-Host "winner ready"
