## 1. Results

**Accuracy including F1score**

-   Pytorch (on NVIDIA A6000)

```bash
Calculating metric for List: ./data/culane/list/test.txt
iou thr: 0.50, tp: 76730, fp: 11264, fn: 28156,precision: 0.8719912721321909, recall: 0.7315561657418531, f1: 0.7956242223143923

```

-   TensorRT (on NVIDIA A6000)

```bash
Calculating metric for List: ./data/culane/list/test.txt
iou thr: 0.50, tp: 77378, fp: 12938, fn: 27508,precision: 0.8567474201691837, recall: 0.7377343020040806, f1: 0.792799254106003

```
## 2. Methods

**YOLOv8 + FENetV1 Head**




-   YOLOv8 - X model is used
-   Conv2d module are added to neck outputs to match dimensions with FEHeadV1

**Pytorch to TensorRT**

-   Models are seperated as backbone, neck, head block
-   Each block is converted from Pytorch to ONNX
-   Next, each block is converted from ONNX to TensorRT

## Reproduce

**Environments**

-   GPU device : A6000 x 1
-   Docker : [nvcr.io/nvidia/pytorch:22.10-py3](http://nvcr.io/nvidia/pytorch:22.10-py3)
-   python : 3.8
-   pytorch : 1.13.0a0+d0d6b1f
-   numpy : 1.24.4
-   mmyolo : 0.6.0
-   mmcv : 2.0.0rc4
-   mmengine : 0.7.1
-   tensorrt : 8.5.0.12

**Reproduce Guide**

1.  Training Pytorch Model
    
    -   Model is trained for 30 epochs with batch size 24 without pretrained model
    -   Run command
    
    ```bash
    $ python main.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py --gpus 0
    
    ```
    
2.  Test Pytorch Model
    
    -   Model is tested with batch size 1 with checkpoint of epoch 26
    -   Add “—view” if you want to save visualized detection output
    -   Run command
    
    ```bash
    $ python main.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py --load_from work_dirs/yolov8_culane/fenetv1/20241107_133400_lr_6e-04_b_24_ckpt_best/ckpt/26_best.pth --test --view --gpus 0
    
    ```
    
3.  Export from Pytorch to TensorRT
    
    -   Run command
    
    ```bash
    $ python export.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py --load_from work_dirs/yolov8_culane/fenetv1/20241107_133400_lr_6e-04_b_24_ckpt_best/ckpt/26_best.pth --test --gpus 0
    $ trtexec --onnx=yolov8_backbone_culane.onnx --saveEngine=yolov8_backbone_culane.trt
    $ trtexec --onnx=yolov8_neck_culane.onnx --saveEngine=yolov8_neck_culane.trt
    $ trtexec --onnx=yolov8_heads_culane.onnx --saveEngine=yolov8_heads_culane.trt
    
    ```
    
4.  Quantization
    
5.  Test TensorRT Model
    
    -   nvtx code is added to check inference time accurately
    -   Run command
    
    ```bash
    $ nsys profile -o yolov8_inference_culane.qdrep \\
    python run_trt.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py \\
    --load_from work_dirs/yolov8_culane/fenetv1/20241107_133400_lr_6e-04_b_24_ckpt_best/ckpt/26_best.pth \\
    --backbone_trt_path yolov8_backbone_culane.trt \\
    --neck_trt_path yolov8_neck_culane.trt \\
    --heads_trt_path yolov8_heads_culane.trt \\
    --test --gpus 0
    
    ```
