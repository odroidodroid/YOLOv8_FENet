import argparse
import os
import time
from signal import SIG_DFL, SIGPIPE, signal

import numpy as np
import onnx
import torch
import torch.backends.cudnn as cudnn
import torch_tensorrt
import tqdm

from fenet.datasets import build_dataloader
from fenet.engine.runner import Runner
from fenet.models.registry import build_net
from fenet.utils.config import Config
from fenet.utils.net_utils import load_network, resume_network, save_model
from fenet.utils.recorder import build_recorder


def quantize_input(input, cfg) :
    q_input = torch.tensor(cfg.img_norm.std).reshape(3, -1) * input 
    q_input = torch.tensor(cfg.img_norm.mean).reshape(3, -1) + q_input
    
    q_input = torch.clamp(q_input - 128.0, -127.0, 128.0)
    return q_input

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    recorder = build_recorder(cfg)
    net = build_net(cfg).cuda()
    
    recorder.logger.info("Network : \n" + str(net))
    load_network(net, cfg.load_from)
    test_loader = build_dataloader(cfg.dataset.test,
                                   cfg,
                                   is_train=False)
    net.eval()
    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(test_loader,
                                                         cache_file='./calibration.cache',
                                                         use_cache=False,
                                                         algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                                                         device='cuda')
    onnx_model = torch.onnx.export(net.backbone,
                                   torch.randn(1, 3, 320, 800).cuda(),
                                   './yolov8_backbone_culane.onnx',
                                   export_params=True,
                                   opset_version=11,
                                   do_constant_folding=True,
                                   input_names=['input'],
                                   output_names=['output0','output1','output2'],
                                   dynamic_axes={
                                       'input' : {0 : 'batch_size'},
                                       'output0' : {0 : 'batch_size'},
                                       'output1' : {0 : 'batch_size'},
                                       'output2' : {0 : 'batch_size'},
                                   })

    onnx_model = torch.onnx.export(net.neck,
                                   [torch.randn(1, 320, 40, 100).cuda(),torch.randn(1, 640, 20, 50).cuda(),torch.randn(1, 640, 10, 25).cuda()],
                                   './yolov8_neck_culane.onnx',
                                   export_params=True,
                                   opset_version=11,
                                   do_constant_folding=True,
                                   input_names=['input0','input1','input2'],
                                   output_names=['output0','output1', 'output2'],
                                   dynamic_axes={
                                       'input0' : {0 : 'batch_size'},
                                       'input1' : {0 : 'batch_size'},
                                       'input2' : {0 : 'batch_size'},
                                       'output0' : {0 : 'batch_size'},
                                       'output1' : {0 : 'batch_size'},
                                       'output2' : {0 : 'batch_size'},
                                   })
    onnx_model = torch.onnx.export(net.heads,
                                   [torch.randn(1, 64, 40, 100).cuda(),torch.randn(1, 64, 20, 50).cuda(),torch.randn(1, 64, 10, 25).cuda()],
                                   './yolov8_heads_culane.onnx',
                                   export_params=True,
                                   opset_version=16,
                                   do_constant_folding=True,
                                   input_names=['input0','input1','input2'],
                                   output_names=['output'],
                                   dynamic_axes={
                                       'input0' : {0 : 'batch_size'},
                                       'input1' : {0 : 'batch_size'},
                                       'input2' : {0 : 'batch_size'},
                                       'output' : {0 : 'batch_size'},
                                   })

    # run command
    # trtexec --onnx=yolov8_backbone_culane.onnx --saveEngine=yolov8_backbone_culane.trt
    # trtexec --onnx=yolov8_neck_culane.onnx --saveEngine=yolov8_neck_culane.trt
    # trtexec --onnx=yolov8_heads_culane.onnx --saveEngine=yolov8_heads_culane.trt

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument(
        '--test',
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
