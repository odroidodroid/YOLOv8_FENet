nsys profile -o yolov8_inference_culane.qdrep \
python run_trt.py configs/yolov8/FENetV1_culane_yolov8_x_mask-refine_syncbn_fast_8xb16-500e.py \
--load_from work_dirs/yolov8_culane/fenetv1/20241107_133400_lr_6e-04_b_24_ckpt_best/ckpt/26_best.pth \
--backbone_trt_path yolov8_backbone_culane.trt \
--neck_trt_path yolov8_neck_culane.trt \
--heads_trt_path yolov8_heads_culane.trt \
--test --gpus 0