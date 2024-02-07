import logging
from models.classifier import Classifier, NormalizedClassifier
from models.img_resnet import Benchmark
from models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50


__factory = {
     'resnet50': Benchmark,
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}


__model_factory = ['resnet18',  'resnet34',  'resnet50',  'resnet101',  'resnet152',  'resnext50_32x4d',  'resnext101_32x8d',  'resnet50_fc512',  'se_resnet50',  'se_resnet50_fc512',  'se_resnet101',  'se_resnext50_32x4d',  'se_resnext101_32x4d',  'densenet121',  'densenet169',  'densenet201',  'densenet161',  'densenet121_fc512',  'inceptionresnetv2',  'inceptionv4',  'xception',  'resnet50_ibn_a',  'resnet50_ibn_b',  'nasnsetmobile',  'mobilenetv2_x1_0',  'mobilenetv2_x1_4',  'shufflenet',  'squeezenet1_0',  'squeezenet1_0_fc512',  'squeezenet1_1',  'shufflenet_v2_x0_5',  'shufflenet_v2_x1_0',  'shufflenet_v2_x1_5',  'shufflenet_v2_x2_0',  'mudeep',  'resnet50mid',  'hacnn',  'pcb_p6',  'pcb_p4',  'mlfn',  'osnet_x1_0',  'osnet_x0_75',  'osnet_x0_5',  'osnet_x0_25',  'osnet_ibn_x1_0',  'osnet_ain_x1_0',  'osnet_ain_x0_75',  'osnet_ain_x0_5',  'osnet_ain_x0_25']

def build_model(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')
    # Build backbone
    logger.info("Initializing model: {}".format(config.MODEL.NAME))
         
    if config.MODEL.NAME in __model_factory:
        model = Benchmark(config)
    else:
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    logger.info("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    else:
        identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)

    clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)

    return model, identity_classifier, clothes_classifier