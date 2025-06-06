import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b3

def build_efficientnet_classifier(version='b3', num_classes=3):
    """
    EfficientNet-b0 또는 b3 기반의 커스텀 분류기 생성.
    version: 'b0' 또는 'b3'
    """
    if version == 'b3':
        model = efficientnet_b3(weights=None)
    elif version == 'b0':
        model = efficientnet_b0(weights=None)
    else:
        raise ValueError("지원하지 않는 EfficientNet version입니다: {}".format(version))
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model

def load_model(model_path, device, version='b3', num_classes=3):
    """
    모델 버전(b0/b3)에 따라 모델 로드.
    """
    model = build_efficientnet_classifier(version=version, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
