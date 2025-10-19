from .ResNet import *
from .ResNet import resnet18_lora, resnet50_lora
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
from .swin import *

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet18_lora": resnet18_lora,
    "resnet50_lora": resnet50_lora,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
    "swin_t": Swin_T,
    "swin_s": Swin_S,
    "swin_b": Swin_B,
}
