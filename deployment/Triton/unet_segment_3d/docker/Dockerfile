# UNet 3D segmentation Triton image (Python backend)
# This Dockerfile is a relocated version of ../../Dockerfile.unet_segment_3d

FROM nvcr.io/nvidia/tritonserver:24.10-py3

# create model directory in container
RUN mkdir -p /models/unet_segment_3d/1

# install project-specific dependencies
COPY docker/requirements.server.txt /tmp/requirements.server.unet_segment_3d.txt
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN python3 -c "import sys; print('Python', sys.version)" \
 && python3 -m pip install --default-timeout=300 --retries 10 -r /tmp/requirements.server.unet_segment_3d.txt \
 && python3 -c "import torch; print('Installed torch:', torch.__version__)" \
 && rm /tmp/requirements.server.unet_segment_3d.txt

# custom unet_segment_3d model only
COPY model_repository/unet_segment_3d/config.pbtxt /models/unet_segment_3d
COPY model_repository/unet_segment_3d/1/model.py /models/unet_segment_3d/1

# weights (copied into image)
# NOTE: Docker build cannot COPY files outside the build context. Keep weights under this sub-project.
COPY training/weights/model_epoch_1000.pth /models/unet_segment_3d/1/model.pth

ENTRYPOINT [ "tritonserver", "--model-repository=/models" ]
