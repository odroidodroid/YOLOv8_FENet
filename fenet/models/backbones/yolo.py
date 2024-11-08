from mmyolo.models.backbones import YOLOv8CSPDarknet
from fenet.models.registry import BACKBONES

@BACKBONES.register_module
class CUYOLOv8CSPDarknet(YOLOv8CSPDarknet) :
    pass