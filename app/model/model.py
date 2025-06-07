# app/api/model.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b3
from app.config import DEVICE


def build_efficientnet_b0_classifier(num_classes=3):
    """
    EfficientNet-B0 기반 분류기 정의
    """
    model = efficientnet_b0(weights=None)
    # torchvision EfficientNet 구조: classifier = nn.Sequential(Dropout, Linear)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model


def build_efficientnet_b3_classifier(num_classes=3):
    """
    EfficientNet-B3 기반 분류기 정의
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


def load_model(arch: str, model_path: str, device: torch.device):
    """
    주어진 아키텍처(arch)와 가중치 파일(model_path)을 사용해 모델 로드
    arch: 'b0' 또는 'b3'
    model_path: pt 파일 경로
    device: CPU/GPU 디바이스
    """
    # 아키텍처별 모델 생성
    if arch == "b0":
        model = build_efficientnet_b0_classifier(num_classes=3)
    elif arch == "b3":
        model = build_efficientnet_b3_classifier(num_classes=3)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # 가중치 불러오기
    state_dict = torch.load(model_path, map_location=device)
    # merge.ipynb 등에서 {'model': state_dict, ...} 형태일 경우
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model