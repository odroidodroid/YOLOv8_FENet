from mmyolo.models.necks import YOLOv8PAFPN
from fenet.models.registry import NECKS
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig
from typing import List, Union

@NECKS.register_module
class CUYOLOv8PAFPN(YOLOv8PAFPN) :
    def __init__(self,
                 final_channels : List[int],
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) :
        super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                deepen_factor=deepen_factor,
                widen_factor=widen_factor,
                num_csp_blocks=num_csp_blocks,
                freeze_all=freeze_all,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                init_cfg=init_cfg)
        self.final_conv0 = nn.Conv2d(int(out_channels[0]*widen_factor), final_channels[0], kernel_size=1)
        self.final_conv1 = nn.Conv2d(int(out_channels[1]*widen_factor), final_channels[1], kernel_size=1)
        self.final_conv2 = nn.Conv2d(int(out_channels[2]*widen_factor), final_channels[2], kernel_size=1)

    def forward(self, x) :
        outputs = super().forward(x)
        final_outputs = []
        final_outputs.append(self.final_conv0(outputs[0]))
        final_outputs.append(self.final_conv1(outputs[1])) 
        final_outputs.append(self.final_conv2(outputs[2])) 
        return final_outputs