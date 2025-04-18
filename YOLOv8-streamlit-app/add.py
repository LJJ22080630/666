import cv2
import sys
from pathlib import Path


def merge_images(left_img_path, right_img_path, output_path):
    """将两张图像水平拼接并保存到输出路径"""
    # 读取图片
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    # 检查图片是否成功读取
    if left_img is None:
        print(f"错误: 无法读取左侧图片 {left_img_path}")
        return False
    if right_img is None:
        print(f"错误: 无法读取右侧图片 {right_img_path}")
        return False

    # 确保图片高度相同，如果不同则调整右侧图片高度
    if left_img.shape[0] != right_img.shape[0]:
        print("警告: 图片高度不一致，正在调整右侧图片高度...")
        right_img = cv2.resize(right_img, (right_img.shape[1], left_img.shape[0]))

    try:
        # 水平拼接图片
        merged_img = cv2.hconcat([left_img, right_img])

        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存合并后的图片
        success = cv2.imwrite(output_path, merged_img)
        if not success:
            print(f"错误: 无法保存合并后的图片到 {output_path}")
            return False

        print(f"成功: 图片已合并并保存到 {output_path}")
        return True

    except Exception as e:
        print(f"错误: 合并图片时发生异常 - {str(e)}")
        return False


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 4:
        print("用法: python add.py <左侧图片路径> <右侧图片路径> <输出图片路径>")
        sys.exit(1)

    left_path = sys.argv[1]
    right_path = sys.argv[2]
    output_path = sys.argv[3]

    # 执行合并操作
    if not merge_images(left_path, right_path, output_path):
        sys.exit(1)