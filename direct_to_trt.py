import argparse
from torch2trt import torch2trt
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(net, output_name):
    input = torch.zeros(1, 3, 256, 448)
    input_names = ['data']
    output_names = ['features', 'heatmaps', 'pafs']
    model_trt = torch2trt(net, [input], fp16_mode=True)
    torch.save(model_trt.state_dict(), output_name)
    # torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default='models/human-pose-estimation-3d.pth',
                        help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation-3d-trt.pth',
                        help='name of output model in ONNX format')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet(is_convertible_by_mo=True)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    convert_to_onnx(net, args.output_name)
    print('=====================done=====================')
