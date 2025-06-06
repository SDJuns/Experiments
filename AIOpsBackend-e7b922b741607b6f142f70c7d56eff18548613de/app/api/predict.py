from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import torch

from app.inference import disease_inference_sequential
from app.preprocess import preprocess_funcs, disease_names, model_paths
from app.config import DEVICE

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
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
                model_paths,     # 모델 객체를 받아야 함 (경로가 아니라)
                preprocess_funcs,
                disease_names,
                DEVICE
            )
        # 필요하면 여기서 바로 참조 해제
        del image
        torch.cuda.empty_cache()
    except Exception as e:
        print("추론 오류:", e)
        raise HTTPException(status_code=500, detail="추론 과정에서 오류 발생")

    return {"predictions": preds}
