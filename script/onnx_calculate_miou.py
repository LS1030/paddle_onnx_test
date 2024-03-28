import os
import cv2
import numpy as np
import onnxruntime as ort
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

    return parser.parse_args()

def onnx_inference(onnx_session, input_image, mean, std):
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    input_shape = onnx_session.get_inputs()[0].shape
    resized_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
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

def compute_iou_paddle(pred_mask, true_mask, class_id):
    mask = true_mask != 255
    pred_i = np.logical_and(pred_mask == class_id, mask)
    label_i = true_mask == class_id
    intersect_i = np.logical_and(pred_i, label_i)

    # pred_area =np.sum(np.cast(pred_i, "int64"))
    # label_area =np.sum(np.cast(label_i, "int64"))
    # intersect_area =np.sum(np.cast(intersect_i, "int64"))

    pred_area = np.sum(pred_i)
    label_area = np.sum(label_i)
    intersect_area = np.sum(intersect_i)
    union_area = pred_area + label_area - intersect_area

    if union_area == 0:
        iou_score = 1.0  # If there's no overlap, IoU is 1
    else:
        iou_score = intersect_area / union_area
    return iou_score

def compute_iou(pred_mask, true_mask, class_id):
    intersection = np.logical_and(pred_mask == class_id, true_mask == class_id)
    union = np.logical_or(pred_mask == class_id, true_mask == class_id)

    if np.sum(union) == 0:
        iou_score = 1.0  # If there's no overlap, IoU is 1
    else:
        iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def compute_miou(pred_mask, true_mask, num_classes):
    miou_list = []
    iou_dict = {}
    for class_id in range(num_classes):
        class_iou = compute_iou(pred_mask, true_mask, class_id)
        miou_list.append(class_iou)
        iou_dict[class_id] = class_iou
    mIoU = np.mean(miou_list)
    return mIoU, iou_dict

def compute_image_miou(image_path, label_path, onnx_session, mean, std, num_classes):
    # 读取图像和标签
    image = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # 模型推理
    image_infer = onnx_inference(onnx_session, image, mean, std)

    # 调整label尺寸以匹配模型输出形状
    output_shape = onnx_session.get_outputs()[0].shape
    resized_label = cv2.resize(label, (output_shape[3], output_shape[2]))

    # 计算mIoU
    mIoU, iou_dict = compute_miou(image_infer, resized_label, num_classes)

    return mIoU, iou_dict

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
    image_filelist, label_filelist = get_test_filelist(dataset_root_path, filename=image_filelist_path)
    image_number = len(image_filelist)
    print(f"Number of images: {len(image_filelist)}")

    num_classes = 2
    mIou_list = []
    iou_dict_list = {class_id: [] for class_id in range(num_classes)}

    with Progress() as progress:
        task = progress.add_task("Calculating mIoU", total=image_number)
        for image_path, label_path in zip(image_filelist, label_filelist):
            progress.update(task, advance=1)

            miou, iou_dict = compute_image_miou(image_path, label_path, onnx_session, mean, std, num_classes)
            mIou_list.append(miou)
            for class_id, iou_score in iou_dict.items():
                iou_dict_list[class_id].append(iou_score)

    # 输出统计信息
    print("mIoU: {:.2f}%".format(np.mean(mIou_list) * 100))
    print("IoU per class:")
    for class_id in range(num_classes):
        class_iou_list = iou_dict_list[class_id]
        class_iou = np.mean(class_iou_list)
        print("Class {}: {:.2f}%".format(class_id, class_iou * 100))

if __name__ == '__main__':
    args = parse_args()
    main(args)
