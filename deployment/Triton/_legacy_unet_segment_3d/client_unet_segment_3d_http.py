"""Minimal HTTP client for Triton `unet_segment_3d` Python backend.

Sends a NIfTI (.nii.gz) file as BYTES tensor INPUT0 and receives BYTES tensor
OUTPUT0 (another .nii.gz), then writes it to disk.

Prereqs (host python):
  pip install tritonclient[http] numpy

Run:
  python client_unet_segment_3d_http.py --in "E:\\xu\\DataSets\\liulian\\470_333_310_0.3.nii.gz" --out pred_from_triton.nii.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint, host:port")
    ap.add_argument("--model", default="unet_segment_3d")
    ap.add_argument("--in", dest="in_path", required=True, help="Input .nii/.nii.gz path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output .nii.gz path")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    data = in_path.read_bytes()

    # Triton BYTES tensor expects a numpy object array of bytes.
    input0 = np.array([data], dtype=object)

    client = httpclient.InferenceServerClient(url=args.url, verbose=False)

    inputs = [httpclient.InferInput("INPUT0", input0.shape, "BYTES")]
    inputs[0].set_data_from_numpy(input0)

    outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

    result = client.infer(model_name=args.model, inputs=inputs, outputs=outputs)
    out = result.as_numpy("OUTPUT0")
    if out is None:
        raise RuntimeError("No OUTPUT0 returned")

    # out is np.ndarray dtype=object with bytes in element 0
    out_bytes = out.reshape(-1)[0]
    if isinstance(out_bytes, memoryview):
        out_bytes = out_bytes.tobytes()

    out_path.write_bytes(out_bytes)
    print(f"Wrote: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
