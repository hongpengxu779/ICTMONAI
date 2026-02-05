Param(
  [string]$Image = "monai_triton:unet_segment_3d",
  [string]$Url = "localhost"
)

$ErrorActionPreference = "Stop"

Write-Host "Running Triton from image $Image ..."

# NOTE: --shm-size is required for Python backend shared memory when sending BYTES.
docker run --rm --gpus all --shm-size=2g `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  $Image
