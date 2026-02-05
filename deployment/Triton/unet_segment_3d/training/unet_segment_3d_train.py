import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import logging
import os
from pathlib import Path
import sys

from monai.config import print_config
from monai.data import ArrayDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.handlers import (
    MeanDice,
    # MLFlowHandler,  # 需要安装 mlflow: pip install mlflow
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
)
from monai.utils import first


import ignite.engine
import ignite.handlers
import torch


print_config()

directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = r"E:\xu\DataSets\liulian"
print(root_dir)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)



images = sorted(glob.glob(os.path.join(root_dir, "im*.nii.gz")))
segs = sorted(glob.glob(os.path.join(root_dir, "seg*.nii.gz")))

# Define transforms for image and segmentation
imtrans = Compose(
    [
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        RandSpatialCrop((128, 128, 128), random_size=False),
    ]
)
segtrans = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        RandSpatialCrop((128, 128, 128), random_size=False),
    ]
)

# Define nifti dataset, dataloader
ds = ArrayDataset(images, imtrans, segs, segtrans)
loader = DataLoader(ds, batch_size=10, num_workers=0, pin_memory=torch.cuda.is_available())
im, seg = first(loader)
print(im.shape, seg.shape)

# Create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
net = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss = DiceLoss(sigmoid=True)
lr = 1e-3
opt = torch.optim.Adam(net.parameters(), lr)

# Create trainer
trainer = ignite.engine.create_supervised_trainer(net, opt, loss, device, False)

# optional section for checkpoint and tensorboard logging
# adding checkpoint handler to save models (network
# params and optimizer stats) during training
log_dir = os.path.join(root_dir, "logs")
checkpoint_handler = ignite.handlers.ModelCheckpoint(log_dir, "net", n_saved=10, require_empty=False)
trainer.add_event_handler(
    event_name=ignite.engine.Events.EPOCH_COMPLETED,
    handler=checkpoint_handler,
    to_save={"net": net, "opt": opt},
)

# StatsHandler prints loss at every iteration
# user can also customize print functions and can use output_transform to convert
# engine.state.output if it's not a loss value
train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
train_stats_handler.attach(trainer)

# TensorBoardStatsHandler plots loss at every iteration
train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
train_tensorboard_stats_handler.attach(trainer)

# MLFlowHandler plots loss at every iteration on MLFlow web UI
# 注释掉 MLFlow 相关代码（需要安装 mlflow: pip install mlflow）
# mlflow_dir = os.path.join(log_dir, "mlruns")
# train_mlflow_handler = MLFlowHandler(tracking_uri=Path(mlflow_dir).as_uri(), output_transform=lambda x: x)
# train_mlflow_handler.attach(trainer)

# optional section for model validation during training
validation_every_n_epochs = 1
# Set parameters for validation
metric_name = "Mean_Dice"
# add evaluation metric to the evaluator engine
val_metrics = {metric_name: MeanDice()}
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_label = Compose([AsDiscrete(threshold=0.5)])
# Ignite evaluator expects batch=(img, seg) and
# returns output=(y_pred, y) at every iteration,
# user can add output_transform to return other values
evaluator = ignite.engine.create_supervised_evaluator(
    net,
    val_metrics,
    device,
    True,
    output_transform=lambda x, y, y_pred: (
        [post_pred(i) for i in decollate_batch(y_pred)],
        [post_label(i) for i in decollate_batch(y)],
    ),
)

# create a validation data loader
val_imtrans = Compose(
    [
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((128, 128, 128)),
    ]
)
val_segtrans = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((128, 128, 128)),
    ]
)
val_ds = ArrayDataset(images[10:], val_imtrans, segs[10:], val_segtrans)
val_loader = DataLoader(val_ds, batch_size=5, num_workers=0, pin_memory=torch.cuda.is_available())


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
def run_validation(engine):
    evaluator.run(val_loader)


# Add stats event handler to print validation stats via evaluator
val_stats_handler = StatsHandler(
    name="evaluator",
    # no need to print loss value, so disable per iteration output
    output_transform=lambda x: None,
    # fetch global epoch number from trainer
    global_epoch_transform=lambda x: trainer.state.epoch,
)
val_stats_handler.attach(evaluator)

# add handler to record metrics to TensorBoard at every validation epoch
val_tensorboard_stats_handler = TensorBoardStatsHandler(
    log_dir=log_dir,
    # no need to plot loss value, so disable per iteration output
    output_transform=lambda x: None,
    # fetch global epoch number from trainer
    global_epoch_transform=lambda x: trainer.state.epoch,
)
val_tensorboard_stats_handler.attach(evaluator)

# add handler to record metrics to MLFlow at every validation epoch
# 注释掉 MLFlow 相关代码（需要安装 mlflow: pip install mlflow）
# val_mlflow_handler = MLFlowHandler(
#     tracking_uri=Path(mlflow_dir).as_uri(),
#     # no need to plot loss value, so disable per iteration output
#     output_transform=lambda x: None,
#     # fetch global epoch number from trainer
#     global_epoch_transform=lambda x: trainer.state.epoch,
# )
# val_mlflow_handler.attach(evaluator)

# add handler to draw the first image and the corresponding
# label and model output in the last batch
# here we draw the 3D output as GIF format along Depth
# axis, at every validation epoch
val_tensorboard_image_handler = TensorBoardImageHandler(
    log_dir=log_dir,
    batch_transform=lambda batch: (batch[0], batch[1]),
    output_transform=lambda output: output[0],
    global_iter_transform=lambda x: trainer.state.epoch,
)
evaluator.add_event_handler(
    event_name=ignite.engine.Events.EPOCH_COMPLETED,
    handler=val_tensorboard_image_handler,
)

# create a training data loader
train_ds = ArrayDataset(images[:10], imtrans, segs[:10], segtrans)
train_loader = DataLoader(
    train_ds,
    batch_size=5,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

max_epochs = 10
state = trainer.run(train_loader, max_epochs)


import copy
import glob
import os
import torch
from monai.config import print_config
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandSpatialCropSamplesd, RandFlipd, RandRotate90d, Compose
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# 1. 基本配置
print_config()
set_determinism(42)

root_dir = r"E:\xu\DataSets\liulian"   # 替换成你的路径
check_point = r"E:\xu\CT\MONAI\3d_segmentation\checkpoints"
images = sorted(glob.glob(os.path.join(root_dir, "im*.nii.gz")))
labels = sorted(glob.glob(os.path.join(root_dir, "seg*.nii.gz")))
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

# 划分数据集
train_files = data_dicts[:10]
val_files = data_dicts[10:]

# 2. Transform
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys="image"),
    RandSpatialCropSamplesd(
        keys=["image", "label"],
        roi_size=(128, 128, 128),
        num_samples=4,
        random_center=True,
        random_size=False,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys="image"),
])

# 3. Dataset & DataLoader
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

# 4. 网络/损失/优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_func = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(net.parameters(), 1e-3)
dice_metric = DiceMetric(include_background=True, reduction="mean")

# 5. 训练与验证主循环
max_epochs = 1000
val_interval = 10


# 放在循环外
best_metric = -1
best_metric_epoch = -1
best_model_wts = copy.deepcopy(net.state_dict())

for epoch in range(max_epochs):
    print("-" * 10, f"epoch {epoch+1}", "-" * 10)
    net.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        # [B, N, C, D, H, W] -> [B*N, C, D, H, W]
        images = batch_data["image"].reshape(-1, 1, 128, 128, 128).to(device)
        labels = batch_data["label"].reshape(-1, 1, 128, 128, 128).to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step += 1
    print(f"Train average loss: {epoch_loss / step:.4f}")

    # -------- 验证 --------
    if (epoch + 1) % val_interval == 0:
        net.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)        # [1, 1, D, H, W]
                val_labels = val_data["label"].to(device)
                # 自动适配roi_size，保证不会超过体积shape
                input_shape = val_inputs.shape[2:]  # (D, H, W)
                roi_size = tuple(min(r, s) for r, s in zip((128,128,128), input_shape))
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size=roi_size, sw_batch_size=1, predictor=net
                )
                val_outputs = torch.sigmoid(val_outputs)
                val_outputs = (val_outputs > 0.5).float()
                dice_metric(y_pred=val_outputs, y=val_labels)
            # 兼容各种MONAI版本，无"count"属性
            result = dice_metric.aggregate()
            dice_metric.reset()
            if result is not None:
                mean_dice = result.item()
                print(f"Validation Mean Dice: {mean_dice:.4f}")
                # ---------- 保存最新权重 ----------
                model_save_path = os.path.join(root_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict()}, model_save_path)
                print(f"Saved latest model to {model_save_path}")
                # ---------- 保存最佳权重 ----------
                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(net.state_dict())
                    torch.save(best_model_wts, os.path.join(root_dir, "best_model.pth"))
                    print(f"Saved best model at epoch {best_metric_epoch}, Dice: {best_metric:.4f}")
            else:
                print("No samples in validation.")

print("训练完成！")