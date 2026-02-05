#!/bin/bash
# Build Triton container image for UNet 3D segmentation model only.

set -e

demo_app_image_name="monai_triton:unet_segment_3d"

docker build -t ${demo_app_image_name} -f Dockerfile.unet_segment_3d .
