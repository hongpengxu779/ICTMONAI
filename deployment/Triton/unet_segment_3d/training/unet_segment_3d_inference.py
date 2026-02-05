# 解决 OMP: Error #15 - 多个 OpenMP 运行时库冲突问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose, LoadImage, ScaleIntensity, EnsureChannelFirst, Activations, AsDiscrete
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
import time

# 路径设置
root_dir = r"E:\xu\DataSets\liulian"
check_points = r"E:\xu\CT\MONAI\3d_segmentation\checkpoints\model_epoch_1000.pth"
model_path = os.path.join(check_points)
output_dir = os.path.join(root_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)

input_path = os.path.join(root_dir, "470_333_310_0.3.nii.gz")

# 加载和预处理（保持原始尺寸）
transform = Compose([
    LoadImage(image_only=True),
    ScaleIntensity(),
    EnsureChannelFirst(),
])

input_img = transform(input_path)  # shape: (1, D, H, W)
input_tensor = input_img.unsqueeze(0).to(torch.float32)  # shape: (1, 1, D, H, W)

# 构建网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# 加载模型
checkpoint = torch.load(model_path, map_location=device)
net.load_state_dict(checkpoint["net"])
net.eval()

# 后处理
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.0005)])

# 推理与保存（滑窗）
roi_size = (128, 128, 128)  # 与训练时保持一致
sw_batch_size = 1

with torch.no_grad():
    start_time = time.time()
    output = sliding_window_inference(
        input_tensor.to(device),
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=net,
    )
    output_post = post_pred(output)
    output_np = output_post.cpu().numpy()[0, 0]
    end_time = time.time()
    print("推理时间：", end_time - start_time)

    # 保存结果
    affine = nib.load(input_path).affine
    save_path = os.path.join(output_dir, "pred_single.nii.gz")
    nib.save(nib.Nifti1Image(output_np.astype(np.uint8), affine), save_path)
    print(f"✅ 推理完成，结果已保存：{save_path}")

