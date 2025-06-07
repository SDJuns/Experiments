# app/api/predict.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import torch

from app.config import DEVICE, MODEL_CONFIG
from app.model import load_model
from app.preprocess import preprocess_funcs, disease_keys, disease_names
from app.inference import disease_inference_sequential

router = APIRouter()

# 서버 시작 시, MODEL_CONFIG 기반으로 모든 모델을 미리 로드
loaded_models = {}
for disease, conf in MODEL_CONFIG.items():
    loaded_models[disease] = load_model(
        conf["arch"],
        str(conf["path"]),
        DEVICE
    )

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 파일 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="올바른 이미지 파일이 아닙니다.")

    try:
        with torch.no_grad():
            preds = disease_inference_sequential(
                image,
                loaded_models,      # 모델 객체 딕셔너리 전달
                preprocess_funcs,
                disease_keys,
                DEVICE
            )
        # 참조 해제 및 캐시 정리
        del image
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print("추론 오류:", e)
        raise HTTPException(status_code=500, detail="추론 과정에서 오류 발생")

    return {"predictions": preds}