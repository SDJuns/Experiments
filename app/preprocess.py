# app/preprocess.py

from torchvision import transforms
from app.config import MODEL_CONFIG

# 1. 질환 키 리스트 (config.py의 KEY와 일치해야 합니다)
disease_keys = list(MODEL_CONFIG.keys())  # 예: ["dandruff", "hair_loss", ...]

# 2. 사용자에게 보여줄 질환명 리스트 (원한다면 한글로 매핑)
#    disease_keys 순서와 1:1 매핑되어야 함
disease_names = disease_keys  # 필요시 한글 이름 리스트로 대체 가능

# 3. 공통 전처리 함수 (모든 질환에 동일 적용)
default_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. 모든 질환에 동일한 전처리 적용
dtype = len(disease_keys)
preprocess_funcs = [default_transform for _ in range(dtype)]

# 
# 이전의 model_paths 리스트는 더 이상 사용되지 않습니다.
# predict.py에서 loaded_models와 disease_keys, preprocess_funcs를 함께 사용하세요.
