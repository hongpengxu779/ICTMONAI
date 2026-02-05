Param(
  [string]$Tag = "monai_triton:unet_segment_3d",
  [switch]$NoCache
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dockerfile = Join-Path $root "docker\Dockerfile"

$noCacheArg = @()
if ($NoCache) {
  $noCacheArg = @("--no-cache")
}

Write-Host "Building image $Tag ..." -ForegroundColor Cyan
Write-Host ("Docker build cache: {0}" -f ($(if ($NoCache) { "DISABLED (--no-cache)" } else { "ENABLED" })))

docker build @noCacheArg -t $Tag -f $dockerfile $root
