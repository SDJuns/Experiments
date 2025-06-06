# app/api/model.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

def build_efficientnet_b3_classifier(num_classes=3):
    """
    EfficientNet-B3 기반, classifier를 Sequential로 커스텀한 분류기.
    (merge.ipynb에서 학습 및 저장한 pt/pth 파일 구조와 반드시 일치!)
    """
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model

def load_model(model_path, device):
    """
    pt/pth 파일 경로와 device를 받아서 모델을 로드하여 반환.
    파일이 dict 형태라면 'model' 키 내부의 state_dict를 사용.
    """
    model = build_efficientnet_b3_classifier(num_classes=3)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
