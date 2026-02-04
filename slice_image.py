import os
import cv2
import numpy as np
from pathlib import Path


def slice_images(source_img_dir, output_base_dir, slice_size=640, overlap_ratio=0.2):
    """
    只将高分辨率图像切割成带有重叠的切片，不处理标签。

    Args:
        source_img_dir (str): 原始图像文件夹路径
        output_base_dir (str): 输出根目录
        slice_size (int): 切片尺寸（默认640）
        overlap_ratio (float): 重叠比例（0.0 ~ 1.0，默认0.2）
    """

    # 计算重叠像素和步长
    overlap_pixels = int(slice_size * overlap_ratio)
    stride = slice_size - overlap_pixels

    # 创建输出文件夹
    output_img_dir = Path(output_base_dir) / "slice_images"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # 获取图像文件列表
    source_img_path = Path(source_img_dir)
    image_files = [f.name for f in source_img_path.iterdir()
                   if f.suffix.lower() in image_extensions and f.is_file()]

    print(f"找到 {len(image_files)} 张图像待处理。")
    print(f"切片参数: 尺寸={slice_size}, 重叠={overlap_pixels}像素(比例{overlap_ratio}), 步长={stride}")

    total_slices = 0
    saved_slices = 0

    for img_file in image_files:
        # 构建路径
        img_path = source_img_path / img_file
        base_name = img_path.stem

        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图像 {img_file}，跳过。")
            continue

        img_h, img_w = img.shape[:2]
        print(f"处理图像: {img_file} ({img_w}x{img_h})")

        # 生成切片坐标
        y_coords = list(range(0, img_h, stride))
        x_coords = list(range(0, img_w, stride))

        # 确保最后一个切片不会超出图像边界
        if y_coords and y_coords[-1] + slice_size > img_h:
            y_coords[-1] = max(0, img_h - slice_size)
        if x_coords and x_coords[-1] + slice_size > img_w:
            x_coords[-1] = max(0, img_w - slice_size)

        image_slices_count = 0
        saved_image_slices = 0

        # 遍历所有可能的切片位置
        for y in y_coords:
            for x in x_coords:
                # 计算当前切片边界
                x_end = min(x + slice_size, img_w)
                y_end = min(y + slice_size, img_h)

                # 确保切片尺寸正确（边缘情况）
                current_slice_width = x_end - x
                current_slice_height = y_end - y

                if current_slice_width < slice_size or current_slice_height < slice_size:
                    # 创建边缘填充的切片
                    slice_img = np.zeros((slice_size, slice_size, 3), dtype=np.uint8)
                    slice_img[0:current_slice_height, 0:current_slice_width] = img[y:y_end, x:x_end]
                else:
                    slice_img = img[y:y_end, x:x_end]

                total_slices += 1
                image_slices_count += 1

                # 生成切片文件名并保存
                slice_id = f"{base_name}_x{x}_y{y}"
                slice_img_name = f"{slice_id}.jpg"
                slice_img_path = output_img_dir / slice_img_name

                # 保存切片图像
                cv2.imwrite(str(slice_img_path), slice_img)
                saved_slices += 1
                saved_image_slices += 1

        print(f"  生成 {image_slices_count} 个切片，保存 {saved_image_slices} 个")

    print(f"\n处理完成！")
    print(f"总共生成 {total_slices} 个切片")
    print(f"成功保存 {saved_slices} 个切片")
    print(f"图像切片保存在: {output_img_dir}")


def main():
    source_image_directory = r"D:\bupi\2073-0924-07\2073-0924-07"  # 原始大图文件夹
    output_directory = r"D:\bupi\2073-0924-07\2073-0924-07_slicers"  # 输出文件夹

    # 切片参数
    SLICE_SIZE = 640  # 切片尺寸
    OVERLAP_RATIO = 0.2  # 重叠比例（20%）

    # 执行切片
    slice_images(
        source_img_dir=source_image_directory,
        output_base_dir=output_directory,
        slice_size=SLICE_SIZE,
        overlap_ratio=OVERLAP_RATIO
    )


if __name__ == "__main__":
    main()