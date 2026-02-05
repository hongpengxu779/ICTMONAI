# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import logging
import os
import pathlib
import tempfile
from typing import Any, List

import numpy as np
import nibabel as nib
import torch
import torch.backends.cudnn as cudnn

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, EnsureChannelFirst, LoadImage, ScaleIntensity

import triton_python_backend_utils as pb_utils

logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Triton Python backend entrypoint for UNet 3D segmentation."""

    def initialize(self, args):
        # 兼容某些环境的 OpenMP 冲突（按你本地脚本设置保持一致）
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        self.model_config = json.loads(args["model_config"])

        # 设备选择：优先用 Triton 分配的实例设备
        instance_kind = args.get("model_instance_kind", "CPU")
        instance_device_id = args.get("model_instance_device_id", "0")

        self.device = torch.device("cpu")
        if instance_kind == "GPU" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{instance_device_id}")
            cudnn.enabled = True

        # 与你 inference 脚本保持一致的网络结构
        self.net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)

        # 读取权重：按你的偏好，默认从同目录下的 model.pth 加载
        # 同时保留环境变量覆盖能力，方便后续切换
        default_ckpt = str(pathlib.Path(__file__).with_name("model.pth"))
        self.ckpt_path = os.environ.get("UNET_SEG3D_CKPT", default_ckpt)

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"UNet checkpoint not found: {self.ckpt_path}. "
                "Place model.pth next to model.py (recommended) or set env UNET_SEG3D_CKPT."
            )

        checkpoint = torch.load(self.ckpt_path, map_location=self.device)

        # checkpoint 兼容：
        # 1) 直接就是 state_dict
        # 2) {'net': state_dict}
        # 3) {'state_dict': state_dict}
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "net" in checkpoint:
                state_dict = checkpoint["net"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

        self.net.load_state_dict(state_dict)
        self.net.eval()

        self.pre_transform = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                EnsureChannelFirst(),
            ]
        )

        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.0005)])

        # sliding window 参数（与脚本一致，可后续参数化）
        self.roi_size = (128, 128, 128)
        self.sw_batch_size = 1

        logger.info(
            "Initialized unet_segment_3d on %s, ckpt=%s, roi=%s, sw_batch=%s",
            self.device,
            self.ckpt_path,
            self.roi_size,
            self.sw_batch_size,
        )

    # NOTE:
    # Some Triton Python backend versions do not expose `pb_utils.InferenceRequest`.
    # If we use it in type annotations, model import can fail at load time with:
    # AttributeError: module 'triton_python_backend_utils' has no attribute 'InferenceRequest'
    # So we keep the signature untyped (or typed as Any) for compatibility.
    def execute(self, requests: List[Any]):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                input_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
                # client 发送 shape (1, 1) 的 bytes：np.array([[image_bytes]], dtype=np.bytes_)
                in_bytes = input_0.as_numpy().astype(np.bytes_).tobytes()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as f_in:
                    f_in.write(in_bytes)
                    tmp_in_path = f_in.name

                # 预处理 -> (C, D, H, W)
                img = self.pre_transform(tmp_in_path)
                # 变成 (N, C, D, H, W)
                img_t = img.unsqueeze(0).to(self.device, dtype=torch.float32)

                # 滑窗推理
                with torch.no_grad():
                    pred = sliding_window_inference(
                        img_t,
                        roi_size=self.roi_size,
                        sw_batch_size=self.sw_batch_size,
                        predictor=self.net,
                    )
                    pred = self.post_pred(pred)

                # pred -> numpy mask (D,H,W)
                mask_np = pred.detach().cpu().numpy()[0, 0].astype(np.uint8)

                # 复用输入 affine（保持空间信息）
                affine = nib.load(tmp_in_path).affine

                # 写回 nii.gz bytes
                nii = nib.Nifti1Image(mask_np, affine)

                # nibabel expects a filename/path. Write to a temp file then read bytes.
                out_bytes: bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as f_out:
                    tmp_out_path = f_out.name
                try:
                    nib.save(nii, tmp_out_path)
                    with open(tmp_out_path, "rb") as f:
                        out_bytes = f.read()
                finally:
                    try:
                        if "tmp_out_path" in locals() and os.path.exists(tmp_out_path):
                            os.unlink(tmp_out_path)
                    except Exception:
                        pass

                out_arr = np.array([[out_bytes]], dtype=np.bytes_)
                out_tensor = pb_utils.Tensor("OUTPUT0", out_arr)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                logger.exception("unet_segment_3d execute() failed")
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e))))
            finally:
                # 清理临时输入文件
                try:
                    if "tmp_in_path" in locals() and os.path.exists(tmp_in_path):
                        os.unlink(tmp_in_path)
                except Exception:
                    pass

        return responses

    def finalize(self):
        pass
