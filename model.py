import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    # Using mobilenet v2 as a backbone for constructing a light-weight model
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    model = FasterRCNN(backbone, num_classes=num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # converting the model into half precision
    """
    model.half()  
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    """
    return model