import os
import cv2
import onnxruntime as ort
import numpy as np
from rich.progress import Progress
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # Common params
    parser.add_argument(
        "--config",
        help="The path of deploy config file.",
        type=str,
        required=True)
    parser.add_argument(
        '--model_path',
        help='The path of onnx model to be loaded for evaluation.',
        type=str,
        required=True)
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
        help='The save directory of the predicted results.',
        type=str,
        default='output',
        required=None)

    return parser.parse_args()

def onnx_inference(onnx_session, input_image, mean, std):
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    input_shape = onnx_session.get_inputs()[0].shape
    resized_image = cv2.resize(input_image, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image.astype(np.float32) / 255.0
    input_tensor = (resized_image - mean) / std
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

    output = onnx_session.run([output_name], {input_name: input_tensor})[0]
    output = np.squeeze(output)
    output_tensor = np.argmax(output, axis=0)

    # output_image = np.expand_dims(output_tensor, axis=2)
    # output_image = np.concatenate((output_image, output_image, output_image), axis=-1)
    # output_image *= 255

    # merge_image = np.hstack((resized_image * 255, output_image))

    return output_tensor

def get_test_filelist(dataset_root_path, filename="test.txt"):
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

def image_inference(onnx_session, image_path, mean, std):
    output_tensor = onnx_inference(onnx_session, cv2.imread(image_path), mean, std)

    # 这里假设有一些类别颜色映射关系
    class_colors = {
        0: [0, 0, 0],          # 类别0为黑色
        1: [255, 255, 255],    # 类别1为白色
    }

    # 创建一个空白图像来存储可视化的分割结果
    segmentation_image = np.zeros((output_tensor.shape[0], output_tensor.shape[1], 3))

    # 将每个像素根据分割结果着色
    for class_id, color in class_colors.items():
        segmentation_image[output_tensor == class_id] = color

    return segmentation_image

def main(args):
    # 加载ONNX模型
    onnx_model_path = args.model_path
    onnx_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    print(f"Loaded ONNX model from {onnx_model_path}")

    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Loaded config file from {args.config}")
    mean = np.array(config['Deploy']['transforms'][1]['mean'])
    std = np.array(config['Deploy']['transforms'][1]['std'])
    print(f"mean: {mean}")
    print(f"std: {std}")

    # 读取图像
    dataset_root_path = args.image_root_path
    image_filelist_path = args.image_filelist
    image_filelist, _ = get_test_filelist(dataset_root_path, filename=image_filelist_path)
    print(f"Number of images: {len(image_filelist)}")

    # 保存路径
    save_path = args.save_dir
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    with Progress() as progress:
        task = progress.add_task("Onnx inference", total=len(image_filelist))
        for image_path in image_filelist:
            progress.update(task, advance=1)

            output_image = image_inference(onnx_session, image_path, mean, std)
            cv2.imwrite(os.path.join(save_path, os.path.basename(image_path).replace('.jpg', '.png')), output_image)

if __name__ == '__main__':
    args = parse_args()
    main(args)
