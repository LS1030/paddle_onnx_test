import os
import cv2
from rich.progress import Progress
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Merge images with different inference methods.')

    # Common params
    parser.add_argument(
        '--image_root_path',
        dest='image_root_path',
        help='The root path of the filelist of the images to be predicted.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--image_filelist',
        dest='image_filelist',
        help='The file list of the images to be predicted.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The save directory of the merged results.',
        type=str,
        default='output',
        required=None)
    parser.add_argument(
        '--paddle_inference_dir',
        dest='paddle_inference_dir',
        help='The save directory of the paddle predicted results.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--onnx_inference_dir',
        dest='onnx_inference_dir',
        help='The save directory of the onnx predicted results.',
        type=str,
        default=None,
        required=True)
    
    return parser.parse_args()

def get_image_filelist(dataset_root_path, filename="test.txt"):
    test_list_path = os.path.join(dataset_root_path, filename)
    image_filelist = []
    label_filelist = []

    with open(test_list_path, "r") as f:
        for line in f:
            image_path, label_path = line.strip().split()
            if os.path.exists(os.path.join(dataset_root_path, image_path)) and os.path.exists(os.path.join(dataset_root_path, label_path)):
                image_filelist.append(os.path.join(dataset_root_path, image_path))
                label_filelist.append(os.path.join(dataset_root_path, label_path))

    return image_filelist, label_filelist

def main(args):

    # 读取图像
    dataset_root_path = args.image_root_path
    image_filelist_path = args.image_filelist
    image_filelist, _ = get_image_filelist(dataset_root_path, filename=image_filelist_path)
    print(f"Number of images: {len(image_filelist)}")

    paddle_inference_folder = args.paddle_inference_dir
    onnx_inference_folder = args.onnx_inference_dir
    output_folder = args.save_dir
    print(f"paddle inference folder: {paddle_inference_folder}")
    print(f"onnx inference folder: {onnx_inference_folder}")
    print(f"output folder: {output_folder}")

    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=len(image_filelist))

        for image_file in image_filelist:
            # 构建对应的png文件路径
            image_name = os.path.basename(image_file)
            onnx_label_path = os.path.join(onnx_inference_folder, image_name.replace('.jpg', '.png'))
            paddle_label_path = os.path.join(paddle_inference_folder, image_name.replace('.jpg', '.png'))

            # 更新进度条
            progress.update(task, advance=1, description=f"Processing {image_name}")

            # 读取图片和标签
            if os.path.exists(onnx_label_path) and os.path.exists(paddle_label_path):
                image = cv2.imread(image_file)
                onnx_label = cv2.imread(onnx_label_path)
                paddle_label = cv2.imread(paddle_label_path)

                resize_image = cv2.resize(image, (onnx_label.shape[1], onnx_label.shape[0]))
            else:
                if os.path.exists(onnx_label_path) is False:
                    print(f'{onnx_label_path} not found')
                if os.path.exists(paddle_label_path) is False:
                    print(f'{paddle_label_path} not found')
                continue

            # 检查图片是否成功读取
            if image is not None and onnx_label is not None and paddle_label is not None:
                # print(image.shape)
                # 水平拼接图片
                combined_image = cv2.hconcat([resize_image, paddle_label, onnx_label])

                # 构建输出文件路径
                output_file = os.path.join(output_folder, image_name.replace('.jpg', '.png'))

                # 保存拼接后的图片
                cv2.imwrite(output_file, combined_image)

if __name__ == "__main__":
    args = parse_args()
    main(args)