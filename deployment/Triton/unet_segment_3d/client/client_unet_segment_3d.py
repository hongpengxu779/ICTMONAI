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

import argparse
import os
import sys
import time
from uuid import uuid4

import numpy as np
import nibabel as nib
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def main():
    parser = argparse.ArgumentParser(description="Triton client for UNet 3D segmentation (nii.gz bytes in/out)")
    parser.add_argument("input", type=str, help="Path to a .nii or .nii.gz file")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for predicted mask nii.gz (default: <input>_pred.nii.gz)",
    )
    parser.add_argument("--url", type=str, default="localhost:8000", help="Triton HTTP endpoint")
    args = parser.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"Input not found: {in_path}")
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        base = in_path
        if base.endswith(".nii.gz"):
            base = base[: -len(".nii.gz")]
        elif base.endswith(".nii"):
            base = base[: -len(".nii")]
        out_path = base + "_pred.nii.gz"

    model_name = "unet_segment_3d"

    with open(in_path, "rb") as f:
        image_bytes = f.read()

    input0_data = np.array([[image_bytes]], dtype=np.bytes_)

    inputs = [httpclient.InferInput("INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype))]
    inputs[0].set_data_from_numpy(input0_data)

    outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

    with httpclient.InferenceServerClient(args.url) as client:
        t0 = time.time()
        resp = client.infer(model_name, inputs, request_id=str(uuid4().hex), outputs=outputs)
        dt_ms = (time.time() - t0) * 1000

    out_bytes = resp.as_numpy("OUTPUT0").astype(np.bytes_).tobytes()

    # 直接把 bytes 保存为 nii.gz
    with open(out_path, "wb") as f:
        f.write(out_bytes)

    # 读一下确认能被 nibabel 打开（可选）
    try:
        _ = nib.load(out_path)
    except Exception:
        pass

    print(f"Saved: {out_path} (inference {dt_ms:.0f} ms)")


if __name__ == "__main__":
    main()
