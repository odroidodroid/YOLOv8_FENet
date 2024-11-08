# Contents

### 1. Results

- Accuracy
- Execution speed

### 2. Methods

- YOLOv8 + FENetV1 Head
- Pytorch to TensorRT

### 3. Reproduce

- Environments
- Reproduce Guide

### 4. Limitation

---

### 1. Results

**Accuracy including F1score**

- Pytorch (on NVIDIA A6000)

```bash
Calculating metric for List: ./data/culane/list/test.txt
iou thr: 0.50, tp: 76730, fp: 11264, fn: 28156,precision: 0.8719912721321909, recall: 0.7315561657418531, f1: 0.7956242223143923
```

- TensorRT (on NVIDIA A6000)

```bash
Calculating metric for List: ./data/culane/list/test.txt
iou thr: 0.50, tp: 77378, fp: 12938, fn: 27508,precision: 0.8567474201691837, recall: 0.7377343020040806, f1: 0.792799254106003
```

**Execution speed**

- Pytorch (on NVIDIA A6000)
    - avg : 15.912ms per 1 batch (data is already allocated to GPU)
- TensorRT (on NVIDIA A6000)
    - avg : 76.4758ms per 1 batch (including time allocating and free GPU memory)
    - TensorRT model execution time is only avg 9.515ms (**41% speedup**)

![nsys_tensorrt_detail.PNG](https://prod-files-secure.s3.us-west-2.amazonaws.com/ec6ce208-fc3c-49cd-bdf6-0b69f0fec57b/329d6b85-937d-46c4-b7c1-05fde0ee5ed9/nsys_tensorrt_detail.png)

- orange block is TensorRT execution time of backbone, neck, head model blocks

## 2. Methods

**YOLOv8 + FENetV1 Head**

![토르드라이브_모델_아키텍처.PNG](https://prod-files-secure.s3.us-west-2.amazonaws.com/ec6ce208-fc3c-49cd-bdf6-0b69f0fec57b/4333574d-0301-4fc4-a2ff-5147d24e95f0/%ED%86%A0%EB%A5%B4%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C_%EB%AA%A8%EB%8D%B8_%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98.png)

- YOLOv8 - X model is used
- Conv2d module are added to neck outputs to match dimensions with FEHeadV1

**Pytorch to TensorRT**

- Models are seperated as backbone, neck, head block
- Each block is converted from Pytorch to ONNX
- Next, each block is converted from ONNX to TensorRT

## Reproduce

**Environments**

- GPU device : A6000 x 1
- Docker : nvcr.io/nvidia/pytorch:22.10-py3
- python : 3.8
- pytorch : 1.13.0a0+d0d6b1f
- numpy : 1.24.4
- mmyolo : 0.6.0
- mmcv : 2.0.0rc4
- mmengine : 0.7.1
- tensorrt : 8.5.0.12

**Reproduce Guide**

1. Training Pytorch Model
    - Model is trained for 30 epochs with batch size 24 without pretrained model
    - Run command
    
    ```bash
    $ python main.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py --gpus 0
    ```
    
2. Test Pytorch Model
    - Model is tested with batch size 1 with checkpoint of epoch 26
    - Add “—view” if you want to save visualized detection output
    - Run command
    
    ```bash
    $ python main.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py --load_from work_dirs/yolov8_culane/fenetv1/20241107_133400_lr_6e-04_b_24_ckpt_best/ckpt/26_best.pth --test --view --gpus 0
    ```
    
3. Export from Pytorch to TensorRT
    - Run command
    
    ```bash
    $ python export.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py --load_from work_dirs/yolov8_culane/fenetv1/20241107_133400_lr_6e-04_b_24_ckpt_best/ckpt/26_best.pth --test --gpus 0
    $ trtexec --onnx=yolov8_backbone_culane.onnx --saveEngine=yolov8_backbone_culane.trt
    $ trtexec --onnx=yolov8_neck_culane.onnx --saveEngine=yolov8_neck_culane.trt
    $ trtexec --onnx=yolov8_heads_culane.onnx --saveEngine=yolov8_heads_culane.trt
    ```
    
4. Quantization
5. Test TensorRT Model
    - nvtx code is added to check inference time accurately
    - Run command
    
    ```bash
    $ nsys profile -o yolov8_inference_culane.qdrep \
    python run_trt.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py \
    --load_from work_dirs/yolov8_culane/fenetv1/20241107_133400_lr_6e-04_b_24_ckpt_best/ckpt/26_best.pth \
    --backbone_trt_path yolov8_backbone_culane.trt \
    --neck_trt_path yolov8_neck_culane.trt \
    --heads_trt_path yolov8_heads_culane.trt \
    --test --gpus 0
    ```
    

## Limitation

- Since I devided model to 3 parts (backbone, neck, head), latency of 1 batch necessarily  included memory allocation & free time.
- It can be resolved by running tensorrt models with c++ code.

## Detailed Results

- all results including test splits and iou threshold

**pytorch model**

```bash
2024-11-07 23:41:06,800 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test0_normal.txt
2024-11-07 23:42:43,297 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 30064, fp: 1871, fn: 2713,precision: 0.9414122436198529, recall: 0.9172285444061384, f1: 0.929163060946965
2024-11-07 23:42:43,300 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 29638, fp: 2297, fn: 3139,precision: 0.9280726475653671, recall: 0.9042316258351893, f1: 0.9159970330077882
2024-11-07 23:42:43,304 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 29111, fp: 2824, fn: 3666,precision: 0.9115703773289494, recall: 0.8881532782133813, f1: 0.8997094820126097
2024-11-07 23:42:43,307 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 28304, fp: 3631, fn: 4473,precision: 0.8863002974792548, recall: 0.863532355005034, f1: 0.8747682037334652
2024-11-07 23:42:43,311 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 27159, fp: 4776, fn: 5618,precision: 0.8504462188821043, recall: 0.8285993226957927, f1: 0.8393806403758189
2024-11-07 23:42:43,314 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 25450, fp: 6485, fn: 7327,precision: 0.7969312666353531, recall: 0.7764591024193794, f1: 0.7865619977747559
2024-11-07 23:42:43,317 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 22681, fp: 9254, fn: 10096,precision: 0.7102238922811962, recall: 0.69197913170821, f1: 0.7009828161701076
2024-11-07 23:42:43,321 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 17806, fp: 14129, fn: 14971,precision: 0.5575700641928918, recall: 0.5432467889068554, f1: 0.5503152429224872
2024-11-07 23:42:43,325 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 9783, fp: 22152, fn: 22994,precision: 0.30634100516674495, recall: 0.2984714891539799, f1: 0.30235505006799357
2024-11-07 23:42:43,329 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 1023, fp: 30912, fn: 31754,precision: 0.032033818694222636, recall: 0.03121091008939195, f1: 0.031617010755346765
2024-11-07 23:42:43,329 - fenet.utils.culane_metric - INFO - mean result, total_tp: 221019, total_fp: 98331, total_fn: 106751,precision: 0.6920901831845937, recall: 0.6743112548433352, f1: 0.683085053776734
2024-11-07 23:42:43,396 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test1_crowd.txt
2024-11-07 23:43:55,509 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 20098, fp: 3217, fn: 7905,precision: 0.8620201586961184, recall: 0.7177088169124737, f1: 0.7832729256790989
2024-11-07 23:43:55,512 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 19622, fp: 3693, fn: 8381,precision: 0.8416041175209092, recall: 0.700710638145913, f1: 0.7647219299271211
2024-11-07 23:43:55,515 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 19018, fp: 4297, fn: 8985,precision: 0.8156980484666524, recall: 0.6791415205513694, f1: 0.7411824311157877
2024-11-07 23:43:55,519 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 18242, fp: 5073, fn: 9761,precision: 0.7824147544499249, recall: 0.6514302039067242, f1: 0.7109396313184457
2024-11-07 23:43:55,522 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 17187, fp: 6128, fn: 10816,precision: 0.7371649152905855, recall: 0.6137556690354605, f1: 0.6698234537589149
2024-11-07 23:43:55,525 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 15533, fp: 7782, fn: 12470,precision: 0.6662234612910144, recall: 0.5546905688676214, f1: 0.6053626407888072
2024-11-07 23:43:55,528 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 13022, fp: 10293, fn: 14981,precision: 0.5585245550075059, recall: 0.46502160482805416, f1: 0.5075022409291087
2024-11-07 23:43:55,531 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 9771, fp: 13544, fn: 18232,precision: 0.4190864250482522, recall: 0.3489269006892119, f1: 0.38080205775751197
2024-11-07 23:43:55,534 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 4935, fp: 18380, fn: 23068,precision: 0.21166630924297664, recall: 0.17623111809448988, f1: 0.1923301765462411
2024-11-07 23:43:55,536 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 394, fp: 22921, fn: 27609,precision: 0.01689899206519408, recall: 0.014069921079884298, f1: 0.015355235979578314
2024-11-07 23:43:55,536 - fenet.utils.culane_metric - INFO - mean result, total_tp: 137822, total_fp: 95328, total_fn: 142208,precision: 0.5911301737079133, recall: 0.4921686962111203, f1: 0.5371292723800616
2024-11-07 23:43:55,590 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test2_hlight.txt
2024-11-07 23:44:00,655 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 1076, fp: 217, fn: 609,precision: 0.8321732405259087, recall: 0.6385756676557863, f1: 0.7226326393552719
2024-11-07 23:44:00,656 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 1031, fp: 262, fn: 654,precision: 0.7973704563031709, recall: 0.6118694362017805, f1: 0.6924110141034251
2024-11-07 23:44:00,656 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 962, fp: 331, fn: 723,precision: 0.7440061871616396, recall: 0.570919881305638, f1: 0.64607118871726
2024-11-07 23:44:00,656 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 919, fp: 374, fn: 766,precision: 0.7107501933488012, recall: 0.5454005934718101, f1: 0.6171927468099396
2024-11-07 23:44:00,656 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 848, fp: 445, fn: 837,precision: 0.6558391337973705, recall: 0.5032640949554896, f1: 0.5695097380792478
2024-11-07 23:44:00,656 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 759, fp: 534, fn: 926,precision: 0.5870069605568445, recall: 0.45044510385756675, f1: 0.5097380792478173
2024-11-07 23:44:00,657 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 631, fp: 662, fn: 1054,precision: 0.4880123743232792, recall: 0.3744807121661721, f1: 0.4237743451981196
2024-11-07 23:44:00,657 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 457, fp: 836, fn: 1228,precision: 0.3534416086620263, recall: 0.2712166172106825, f1: 0.30691739422431163
2024-11-07 23:44:00,657 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 227, fp: 1066, fn: 1458,precision: 0.17556071152358854, recall: 0.13471810089020772, f1: 0.1524513096037609
2024-11-07 23:44:00,657 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 31, fp: 1262, fn: 1654,precision: 0.02397525135344161, recall: 0.018397626112759646, f1: 0.020819341840161182
2024-11-07 23:44:00,657 - fenet.utils.culane_metric - INFO - mean result, total_tp: 6941, total_fp: 5989, total_fn: 9909,precision: 0.5368136117556072, recall: 0.41192878338278927, f1: 0.46615177971793154
2024-11-07 23:44:00,660 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test3_shadow.txt
2024-11-07 23:44:08,481 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 2168, fp: 220, fn: 708,precision: 0.9078726968174204, recall: 0.7538247566063978, f1: 0.8237082066869301
2024-11-07 23:44:08,482 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 2123, fp: 265, fn: 753,precision: 0.8890284757118928, recall: 0.7381780250347705, f1: 0.80661094224924
2024-11-07 23:44:08,482 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 2026, fp: 362, fn: 850,precision: 0.8484087102177554, recall: 0.7044506258692629, f1: 0.7697568389057751
2024-11-07 23:44:08,482 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 1827, fp: 561, fn: 1049,precision: 0.7650753768844221, recall: 0.6352573018080667, f1: 0.6941489361702128
2024-11-07 23:44:08,483 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 1707, fp: 681, fn: 1169,precision: 0.714824120603015, recall: 0.5935326842837274, f1: 0.6485562310030395
2024-11-07 23:44:08,483 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 1553, fp: 835, fn: 1323,precision: 0.6503350083752094, recall: 0.5399860917941586, f1: 0.5900455927051673
2024-11-07 23:44:08,483 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 1273, fp: 1115, fn: 1603,precision: 0.5330820770519263, recall: 0.44262865090403336, f1: 0.48366261398176297
2024-11-07 23:44:08,483 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 897, fp: 1491, fn: 1979,precision: 0.3756281407035176, recall: 0.3118915159944367, f1: 0.3408054711246201
2024-11-07 23:44:08,484 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 440, fp: 1948, fn: 2436,precision: 0.18425460636515914, recall: 0.15299026425591097, f1: 0.16717325227963525
2024-11-07 23:44:08,484 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 41, fp: 2347, fn: 2835,precision: 0.017169179229480736, recall: 0.014255910987482615, f1: 0.015577507598784195
2024-11-07 23:44:08,484 - fenet.utils.culane_metric - INFO - mean result, total_tp: 14055, total_fp: 9825, total_fn: 14705,precision: 0.5885678391959799, recall: 0.48869958275382475, f1: 0.5340045592705167
2024-11-07 23:44:08,490 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test4_noline.txt
2024-11-07 23:44:34,684 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 5715, fp: 1961, fn: 8306,precision: 0.7445284002084419, recall: 0.4076028813921974, f1: 0.5268009402221505
2024-11-07 23:44:34,685 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 5567, fp: 2109, fn: 8454,precision: 0.7252475247524752, recall: 0.39704728621353685, f1: 0.5131585011752777
2024-11-07 23:44:34,686 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 5323, fp: 2353, fn: 8698,precision: 0.693460135487233, recall: 0.3796448184865559, f1: 0.49066691247637917
2024-11-07 23:44:34,688 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 5038, fp: 2638, fn: 8983,precision: 0.6563314226159458, recall: 0.35931816560872976, f1: 0.46439599944692816
2024-11-07 23:44:34,689 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 4684, fp: 2992, fn: 9337,precision: 0.6102136529442418, recall: 0.33407032308679835, f1: 0.4317647601050836
2024-11-07 23:44:34,690 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 4155, fp: 3521, fn: 9866,precision: 0.5412975508077124, recall: 0.2963412024819913, f1: 0.38300225837673413
2024-11-07 23:44:34,691 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 3438, fp: 4238, fn: 10583,precision: 0.4478895257946847, recall: 0.24520362313672348, f1: 0.3169101719131677
2024-11-07 23:44:34,692 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 2491, fp: 5185, fn: 11530,precision: 0.32451797811360084, recall: 0.17766207831110478, f1: 0.22961699774162325
2024-11-07 23:44:34,694 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 1244, fp: 6432, fn: 12777,precision: 0.16206357477853048, recall: 0.08872405677198487, f1: 0.11467023090749873
2024-11-07 23:44:34,695 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 121, fp: 7555, fn: 13900,precision: 0.01576341844710787, recall: 0.008629912274445475, f1: 0.011153615707240633
2024-11-07 23:44:34,695 - fenet.utils.culane_metric - INFO - mean result, total_tp: 37776, total_fp: 38984, total_fn: 102434,precision: 0.4921313183949974, recall: 0.2694244347764068, f1: 0.3482140388072084
2024-11-07 23:44:34,717 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test5_arrow.txt
2024-11-07 23:44:44,881 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 2747, fp: 212, fn: 435,precision: 0.9283541737073335, recall: 0.8632935260842237, f1: 0.8946425663572707
2024-11-07 23:44:44,881 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 2721, fp: 238, fn: 461,precision: 0.9195674214261574, recall: 0.85512256442489, f1: 0.8861748900830484
2024-11-07 23:44:44,881 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 2661, fp: 298, fn: 521,precision: 0.8992903007772897, recall: 0.8362664990571967, f1: 0.8666340986809966
2024-11-07 23:44:44,882 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 2589, fp: 370, fn: 593,precision: 0.8749577559986482, recall: 0.8136392206159648, f1: 0.8431851489985345
2024-11-07 23:44:44,882 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 2476, fp: 483, fn: 706,precision: 0.8367691787766137, recall: 0.7781269641734758, f1: 0.8063833251913369
2024-11-07 23:44:44,882 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 2328, fp: 631, fn: 854,precision: 0.786752281176073, recall: 0.7316153362664991, f1: 0.7581827063996092
2024-11-07 23:44:44,882 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 2079, fp: 880, fn: 1103,precision: 0.7026022304832714, recall: 0.6533626649905719, f1: 0.6770884220810944
2024-11-07 23:44:44,883 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 1599, fp: 1360, fn: 1583,precision: 0.5403852652923284, recall: 0.5025141420490258, f1: 0.5207620908646801
2024-11-07 23:44:44,883 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 904, fp: 2055, fn: 2278,precision: 0.30550861777627575, recall: 0.284098051539912, f1: 0.2944145904575802
2024-11-07 23:44:44,883 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 93, fp: 2866, fn: 3089,precision: 0.03142953700574518, recall: 0.029226901319924576, f1: 0.030288226673180263
2024-11-07 23:44:44,883 - fenet.utils.culane_metric - INFO - mean result, total_tp: 20197, total_fp: 9393, total_fn: 11623,precision: 0.6825616762419735, recall: 0.6347265870521683, f1: 0.6577756065787332
2024-11-07 23:44:44,890 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test6_curve.txt
2024-11-07 23:44:48,706 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 787, fp: 176, fn: 525,precision: 0.8172377985462098, recall: 0.5998475609756098, f1: 0.6918681318681319
2024-11-07 23:44:48,707 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 750, fp: 213, fn: 562,precision: 0.778816199376947, recall: 0.5716463414634146, f1: 0.6593406593406593
2024-11-07 23:44:48,707 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 707, fp: 256, fn: 605,precision: 0.7341640706126688, recall: 0.5388719512195121, f1: 0.6215384615384616
2024-11-07 23:44:48,707 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 645, fp: 318, fn: 667,precision: 0.6697819314641744, recall: 0.4916158536585366, f1: 0.567032967032967
2024-11-07 23:44:48,707 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 539, fp: 424, fn: 773,precision: 0.5597092419522326, recall: 0.4108231707317073, f1: 0.47384615384615386
2024-11-07 23:44:48,707 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 395, fp: 568, fn: 917,precision: 0.4101765316718588, recall: 0.3010670731707317, f1: 0.34725274725274724
2024-11-07 23:44:48,707 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 245, fp: 718, fn: 1067,precision: 0.2544132917964694, recall: 0.18673780487804878, f1: 0.2153846153846154
2024-11-07 23:44:48,708 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 113, fp: 850, fn: 1199,precision: 0.11734164070612668, recall: 0.0861280487804878, f1: 0.09934065934065933
2024-11-07 23:44:48,708 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 26, fp: 937, fn: 1286,precision: 0.02699896157840083, recall: 0.019817073170731708, f1: 0.022857142857142854
2024-11-07 23:44:48,708 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 1, fp: 962, fn: 1311,precision: 0.0010384215991692627, recall: 0.0007621951219512195, f1: 0.0008791208791208791
2024-11-07 23:44:48,708 - fenet.utils.culane_metric - INFO - mean result, total_tp: 4208, total_fp: 5422, total_fn: 8912,precision: 0.43696780893042575, recall: 0.3207317073170731, f1: 0.3699340659340659
2024-11-07 23:44:48,711 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test7_cross.txt
2024-11-07 23:44:50,258 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,259 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,260 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,261 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,262 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,262 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,263 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,264 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,265 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,266 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 0, fp: 1318, fn: 0,precision: 0, recall: 0, f1: 0
2024-11-07 23:44:50,266 - fenet.utils.culane_metric - INFO - mean result, total_tp: 0, total_fp: 13180, total_fn: 0,precision: 0.0, recall: 0.0, f1: 0.0
2024-11-07 23:44:50,269 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test_split/test8_night.txt
2024-11-07 23:45:40,598 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 14075, fp: 2072, fn: 6955,precision: 0.8716789496500899, recall: 0.6692819781264859, f1: 0.75718858433978
2024-11-07 23:45:40,601 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 13733, fp: 2414, fn: 7297,precision: 0.8504985446212919, recall: 0.653019495958155, f1: 0.7387901121661241
2024-11-07 23:45:40,603 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 13248, fp: 2899, fn: 7782,precision: 0.8204620053260667, recall: 0.6299572039942939, f1: 0.7126987115689808
2024-11-07 23:45:40,606 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 12617, fp: 3530, fn: 8413,precision: 0.781383538737846, recall: 0.5999524488825487, f1: 0.6787529924415633
2024-11-07 23:45:40,608 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 11637, fp: 4510, fn: 9393,precision: 0.7206911500588344, recall: 0.5533523537803139, f1: 0.626032224224655
2024-11-07 23:45:40,610 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 10373, fp: 5774, fn: 10657,precision: 0.6424103548646808, recall: 0.49324774132192106, f1: 0.5580331925652958
2024-11-07 23:45:40,612 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 8549, fp: 7598, fn: 12481,precision: 0.5294481947110918, recall: 0.40651450309082265, f1: 0.45990800763913176
2024-11-07 23:45:40,614 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 6020, fp: 10127, fn: 15010,precision: 0.37282467331392827, recall: 0.2862577270565858, f1: 0.323856147618151
2024-11-07 23:45:40,617 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 2848, fp: 13299, fn: 18182,precision: 0.17637951322227038, recall: 0.13542558250118877, f1: 0.15321300804260699
2024-11-07 23:45:40,619 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 296, fp: 15851, fn: 20734,precision: 0.018331578621415743, recall: 0.01407513076557299, f1: 0.015923823869596794
2024-11-07 23:45:40,619 - fenet.utils.culane_metric - INFO - mean result, total_tp: 93396, total_fp: 68074, total_fn: 116904,precision: 0.5784108503127516, recall: 0.4441084165477889, f1: 0.5024396804475887
2024-11-07 23:45:40,658 - fenet.utils.culane_metric - INFO - Calculating metric for List: ./data/culane/list/test.txt
2024-11-07 23:50:04,239 - fenet.utils.culane_metric - INFO - iou thr: 0.50, tp: 76730, fp: 11264, fn: 28156,precision: 0.8719912721321909, recall: 0.7315561657418531, f1: 0.7956242223143923
2024-11-07 23:50:04,256 - fenet.utils.culane_metric - INFO - iou thr: 0.55, tp: 75185, fp: 12809, fn: 29701,precision: 0.8544332568129646, recall: 0.7168258871536716, f1: 0.7796038987971796
2024-11-07 23:50:04,275 - fenet.utils.culane_metric - INFO - iou thr: 0.60, tp: 73056, fp: 14938, fn: 31830,precision: 0.8302384253471827, recall: 0.6965276586007666, f1: 0.7575279966818748
2024-11-07 23:50:04,294 - fenet.utils.culane_metric - INFO - iou thr: 0.65, tp: 70181, fp: 17813, fn: 34705,precision: 0.797565743118849, recall: 0.669116946017581, f1: 0.7277167150559932
2024-11-07 23:50:04,312 - fenet.utils.culane_metric - INFO - iou thr: 0.70, tp: 66237, fp: 21757, fn: 38649,precision: 0.7527445053071801, recall: 0.6315142154338996, f1: 0.6868208212360016
2024-11-07 23:50:04,330 - fenet.utils.culane_metric - INFO - iou thr: 0.75, tp: 60546, fp: 27448, fn: 44340,precision: 0.688069641111894, recall: 0.57725530576054, f1: 0.627810037328909
2024-11-07 23:50:04,347 - fenet.utils.culane_metric - INFO - iou thr: 0.80, tp: 51918, fp: 36076, fn: 52968,precision: 0.5900175011932631, recall: 0.49499456552828786, f1: 0.5383450850269598
2024-11-07 23:50:04,364 - fenet.utils.culane_metric - INFO - iou thr: 0.85, tp: 39154, fp: 48840, fn: 65732,precision: 0.44496215651067117, recall: 0.3733005358198425, f1: 0.4059933637494816
2024-11-07 23:50:04,383 - fenet.utils.culane_metric - INFO - iou thr: 0.90, tp: 20407, fp: 67587, fn: 84479,precision: 0.2319135395595154, recall: 0.19456362145567568, f1: 0.2116030692658648
2024-11-07 23:50:04,401 - fenet.utils.culane_metric - INFO - iou thr: 0.95, tp: 2000, fp: 85994, fn: 102886,precision: 0.022728822419710436, recall: 0.019068321796998647, f1: 0.02073828287017835
2024-11-07 23:50:04,401 - fenet.utils.culane_metric - INFO - mean result, total_tp: 535414, total_fp: 344526, total_fn: 513446,precision: 0.608466486351342, recall: 0.5104723223309118, f1: 0.5551783492326835
```
