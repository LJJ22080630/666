import cv2
import argparse
import os


def convert_to_grayscale(input_path, output_path):
    """将单个图片转换为灰度图"""
    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"无法读取图像: {input_path}")
            return False

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, gray_img)
        print(f"已转换并保存: {output_path}")
        return True
    except Exception as e:
        print(f"处理 {input_path} 时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将彩色图片转换为灰度图')
    parser.add_argument('input_path', help='输入文件路径')
    parser.add_argument('output_path', help='输出文件路径')

    args = parser.parse_args()

    # 只有当输出路径包含目录时才创建目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir:  # 只有当output_dir不是空字符串时才创建
        os.makedirs(output_dir, exist_ok=True)

    if convert_to_grayscale(args.input_path, args.output_path):
        print("转换成功")
    else:
        print("转换失败")


if __name__ == '__main__':
    main()