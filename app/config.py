# app/config.py
from pathlib import Path
import torch

# CUDA(GPU)가 있으면 사용, 없으면 CPU 사용
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 질환별 모델 설정 (pizi 제외)
# key: 질환 식별자, value.path: 다운로드된 pt 파일 경로, arch: 모델 아키텍처
MODEL_CONFIG = {
    "dandruff": {"path": Path("app/model_weight/biddem_B0_compressed.pt"), "arch": "b0"},
    "mise":     {"path": Path("app/model_weight/mise_B0_compressed.pt"),   "arch": "b0"},
    "mono":     {"path": Path("app/model_weight/mono_B0_compressed.pt"),   "arch": "b0"},
    "mosa":     {"path": Path("app/model_weight/mosa_B0_compressed.pt"),   "arch": "b0"},
    "talmo":    {"path": Path("app/model_weight/talmo_B0_compressed.pt"),  "arch": "b0"},
}