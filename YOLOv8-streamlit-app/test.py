import os
import cv2
import argparse
from tqdm import tqdm


def process_image(input_path, output_path, clahe_clip_limit=2.0, clahe_grid_size=(8, 8), median_kernel_size=3):
    """
    对单个图像应用CLAHE和中值滤波

    参数:
        input_path: 输入图像路径
        output_path: 输出图像路径
        clahe_clip_limit: CLAHE的对比度限制
        clahe_grid_size: CLAHE的网格大小
        median_kernel_size: 中值滤波的核大小
    """
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像: {input_path}")

    # 处理图像
    if len(img.shape) == 2:  # 灰度图像
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        processed_img = clahe.apply(img)
        processed_img = cv2.medianBlur(processed_img, median_kernel_size)
    else:  # 彩色图像
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_img)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        l_clahe = clahe.apply(l)
        lab_img = cv2.merge((l_clahe, a, b))
        processed_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 保存图像
    if not cv2.imwrite(output_path, processed_img):
        raise ValueError(f"无法保存图像到: {output_path}")

    print(f"成功处理并保存: {output_path}")


def process_directory(input_dir, output_dir, prefix='', **kwargs):
    """
    处理目录中的所有图像

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        prefix: 输出文件名前缀
        kwargs: 传递给process_image的参数
    """
    # 支持的图像格式
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in extensions]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个图像
    for img_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"{prefix}{img_file}")
        try:
            process_image(input_path, output_path, **kwargs)
        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='图像增强工具：应用CLAHE和中值滤波')
    parser.add_argument('input', help='输入路径（文件或目录）')
    parser.add_argument('output', help='输出路径（文件或目录）')
    parser.add_argument('--prefix', default='', help='输出文件名前缀（仅目录模式）')
    parser.add_argument('--clip_limit', type=float, default=2.0, help='CLAHE对比度限制')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[8, 8],
                        help='CLAHE网格大小（宽 高）')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='中值滤波核大小（必须是奇数）')

    args = parser.parse_args()

    # 处理参数
    kwargs = {
        'clahe_clip_limit': args.clip_limit,
        'clahe_grid_size': tuple(args.grid_size),
        'median_kernel_size': args.kernel_size
    }

    try:
        if os.path.isfile(args.input):
            # 单文件模式
            process_image(args.input, args.output, **kwargs)
        elif os.path.isdir(args.input):
            # 目录模式
            process_directory(args.input, args.output, prefix=args.prefix, **kwargs)
        else:
            raise ValueError("输入路径不存在")

        print("操作完成")
    except Exception as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == '__main__':
    main()