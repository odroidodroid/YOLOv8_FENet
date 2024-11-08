import argparse
import os
import time
from signal import SIG_DFL, SIGPIPE, signal

import numpy as np
import onnx
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.backends.cudnn as cudnn
import torch_tensorrt
from tqdm import tqdm
import onnx
import onnxruntime as ort
import torch.cuda.nvtx as nvtx

from fenet.datasets import build_dataloader
from fenet.engine.runner import Runner
from fenet.models.registry import build_net
from fenet.utils.config import Config
from fenet.utils.net_utils import load_network, resume_network, save_model
from fenet.utils.recorder import build_recorder

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path) :
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime :
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, input) :
    context = engine.create_execution_context()
    
    if not isinstance(input, list) :
        input_list = [input]
    else :
        input_list = input
    d_inputs = []
    h_outputs = []
    d_outputs = []
    bindings = []
    
    for i, h_input in enumerate(input_list) :
        if torch.is_tensor(h_input) :
            h_input = h_input.cpu()
        h_input = np.ascontiguousarray(h_input)
        d_input = cuda.mem_alloc(h_input.nbytes)
        cuda.memcpy_htod_async(d_input, h_input)
        d_inputs.append(d_input)
        bindings.append(int(d_input))
    for i in range(len(input_list), context.engine.num_bindings) :
        output_shape = context.get_binding_shape(i)
        h_output = np.empty(output_shape, dtype=np.float32)
        h_outputs.append(h_output)
        d_output = cuda.mem_alloc(h_output.nbytes)
        d_outputs.append(d_output)
        bindings.append(int(d_output))
    
    stream = cuda.Stream()
    
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    for h_output, d_output in zip(h_outputs, d_outputs) :
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
    
    stream.synchronize()
    
    return h_outputs

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
    cfg.batch_size = 1
    test_loader = build_dataloader(cfg.dataset.test,
                                   cfg,
                                   is_train=False)
    net.eval()
    
    backbone_trt = load_engine(args.backbone_trt_path)
    neck_trt = load_engine(args.neck_trt_path)
    heads_trt = load_engine(args.heads_trt_path)
    
    pred = []
    with torch.no_grad() :
        for i, data in enumerate(tqdm(test_loader, desc=f"Testing")) :
            # enable if you want to generate nvtx time
            # nvtx.range_push(f"Batch {i+1} Inference")
            # torch.cuda.synchronize()
            backbone_output = infer(backbone_trt, data['img'])
            neck_output = infer(neck_trt, backbone_output)
            head_output = infer(heads_trt, neck_output)
            head_output_ = [torch.from_numpy(item_).cuda() for item_ in head_output]
            final_output = net.heads.get_lanes(head_output_[0])
            # torch.cuda.synchronize()
            # nvtx.range_pop()
            pred.extend(final_output)
            if cfg.view :
                test_loader.dataset.view(final_output, data['meta'])
    metric = test_loader.dataset.evaluate(pred, cfg.work_dirs)
    recorder.logger.info('metric : '+ str(metric))
        

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
    parser.add_argument('--backbone_trt_path',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--neck_trt_path',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--heads_trt_path',
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
