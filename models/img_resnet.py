from .Similar_Mask_Generate import SMGBlock
from torch import nn
from torch.nn import init
from models.utils import pooling
import torchreid

class Benchmark(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        model = torchreid.models.build_model(
            name=config.MODEL.NAME,
            num_classes=1,
            loss="softmax",
            pretrained=True
        )
        FEATURE_DIM = model.classifier.in_features
        self.smg = SMGBlock(FEATURE_DIM, 12 * 24)
        base = []
        for n, m in model.named_children():
            if isinstance(m, nn.AdaptiveAvgPool2d):
                break
            else:
                base.append(m)
        #resnet50 = torchvision.models.resnet50(pretrained=True)
        # if config.MODEL.RES4_STRIDE == 1:
        #     resnet50.layer4[0].conv2.stride=(1, 1)
        #     resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*base)
        self.start = 0

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            FEATURE_DIM = FEATURE_DIM * 2
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        
        
        config.defrost()
        config.MODEL.FEATURE_DIM = FEATURE_DIM
        config.freeze()
        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        


    def forward(self, x, cluster = False, epoch = None, y_pred = None):
        x = self.base(x)
        if cluster: return x
        if epoch is not None and epoch >= self.start and self.training:
            dist_matrix = self.smg(x)
        else:
            dist_matrix = None
            
        x = self.globalpooling(x)
        f = x.view(x.size(0), -1)
        f = self.bn(f)

        if self.training:
            return f, dist_matrix
        else:
            return f