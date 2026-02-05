Param(
  [Parameter(Mandatory = $true)]
  [string]$In,

  [Parameter(Mandatory = $true)]
  [string]$Out,

  [string]$Url = "localhost:8000",

  [ValidateSet("http", "simple")]
  [string]$Client = "http"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$clientDir = Join-Path $root "client"

$script = switch ($Client) {
  "http" { "client_unet_segment_3d_http.py" }
  "simple" { "client_unet_segment_3d.py" }
}

$scriptPath = Join-Path $clientDir $script

if (-not (Test-Path $scriptPath)) {
  throw "Client script not found: $scriptPath"
}

Write-Host "Running client: $script" -ForegroundColor Cyan
Write-Host "URL:  $Url"
Write-Host "IN:   $In"
Write-Host "OUT:  $Out"

Push-Location $clientDir
try {
  if ($Client -eq "http") {
    python $scriptPath --in $In --out $Out --url $Url
  }
  else {
    python $scriptPath $In --out $Out --url $Url
  }
}
finally {
  Pop-Location
}
