# unet_segment_3d (Triton Python Backend)

本子目录收拢了 **UNet 3D segmentation** 在 Triton Python Backend 下的全部新增内容：
- Triton 模型仓库（model repository）
- 构建镜像的 Dockerfile 与 server requirements
- 本地启动脚本
- 客户端脚本
- 训练/本地推理脚本（可选）

## 目录结构

```
unet_segment_3d/
  docker/
    Dockerfile
    requirements.server.txt
  model_repository/
    unet_segment_3d/
      config.pbtxt
      1/
        model.py
        # model.pth 由镜像构建时 COPY 进容器；通常不要提交大权重到 git
  client/
    client_unet_segment_3d.py
  scripts/
    build.ps1
    run.ps1
  training/
    unet_segment_3d_train.py
    unet_segment_3d_inference.py
```

## 使用

### 构建镜像

```powershell
cd E:\xu\CT\tutorials\deployment\Triton\unet_segment_3d
.\scripts\build.ps1
```

### 启动 Triton

```powershell
.\scripts\run.ps1
```

> 说明：已默认设置 `--shm-size`，否则传输 NIfTI bytes 时可能出现 shared memory 报错。

### 客户端推理

```powershell
python .\client\client_unet_segment_3d.py "E:\xu\DataSets\liulian\470_333_310_0.3.nii.gz" --out "E:\xu\DataSets\liulian\pred_from_triton.nii.gz" --url localhost:8000
```
