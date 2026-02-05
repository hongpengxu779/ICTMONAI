#!/bin/bash
# Run Triton container locally for UNet 3D segmentation model only.

set -e

demo_app_image_name="monai_triton:unet_segment_3d"

docker run \
  --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --runtime=nvidia \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${demo_app_image_name}
