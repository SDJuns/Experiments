# app/config.py

import torch

# CUDA(GPU)가 있으면 사용, 없으면 CPU 사용
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
