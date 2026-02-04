# MONAI 教程使用指南

本指南整理自 Project MONAI 官方教程库，涵盖 MONAI 在医学图像分析中的应用，包括分割、分类、配准、生成模型、部署等多个方向。

---

## 1. 基础环境依赖

### 推荐安装包

```bash
python -m pip install -U pip
python -m pip install -U matplotlib
python -m pip install -U notebook
```

### 安装全部开发依赖：

```bash
pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt
```

### 在 Colab 上运行：

1. 打开 Jupyter Notebook，点击 "Open in Colab" 按钮
2. 更改运行时类型为 GPU
3. 使用 `!nvidia-smi` 检查 GPU 状态

---

## 2. 教程分类概览

| 分类               | 内容简介                                     |
| ---------------- | ---------------------------------------- |
| **2D分类**         | MedNIST 数据集，基础分类任务演示                     |
| **2D分割**         | UNet + 合成数据集，字典式与数组式两种实现                 |
| **3D分类/回归**      | DenseNet3D + IXI 脑数据集                    |
| **3D分割**         | UNet、UNETR、VISTA，使用 BTCV/Brats/Spleen 数据 |
| **图像配准**         | 支持配对与非配对图像注册，含 VoxelMorph 框架             |
| **交互分割**         | DeepEdit、DeepGrow，支持点击/用户引导输入            |
| **部署推理**         | Triton、BentoML、Ray 等服务部署示例               |
| **实验管理**         | MLFlow、Aim、ClearML 接入教程                  |
| **联邦学习**         | 支持 NVFlare、OpenFL、Substra 框架             |
| **数字病理**         | Whole Slide Image 处理与 MIL 分类             |
| **加速技巧**         | CacheDataset、AMP、TensorRT、ThreadBuffer   |
| **Auto3DSeg**    | 自动分割模型搜索与训练                              |
| **自监督学习**        | ViT-UNETR 自监督训练 + 微调                     |
| **生成模型**         | LDM、VAE-GAN、SPADE，支持合成/图像翻译              |
| **Transform 工具** | 各类 transforms 与 postprocessing 演示        |

---

## 3. 推荐重点教程

| 教程                                      | 说明                       |
| --------------------------------------- | ------------------------ |
| `spleen_segmentation_3d.ipynb`          | 经典 UNet + Spleen 分割任务全流程 |
| `unetr_btcv_segmentation_3d.ipynb`      | 使用 UNETR 在 BTCV 上进行多器官分割 |
| `fast_training_tutorial.ipynb`          | MONAI 加速训练技巧合集           |
| `TensorRT_inference_acceleration.ipynb` | 使用 TensorRT 进行推理加速       |
| `auto3dseg/`                            | 自动 3D 分割任务探索与定制          |
| `spleen_segmentation_aim.ipynb`         | 使用 Aim 进行实验可视化管理         |

---

## 4. 支持与交流

* 📚 官方文档：[https://docs.monai.io](https://docs.monai.io)
* 💬 问题与讨论：[Discussions](https://github.com/Project-MONAI/MONAI/discussions)
* 🐞 Bug反馈：[Issues](https://github.com/Project-MONAI/MONAI/issues)

---

如需 MONAI 某个特定模块的中文讲解与定制案例，请联系项目维护者或提出具体需求。
