import torch
from app.model.model import load_model
from app.recommendation.utils import get_recommendations_by_disease

# 1. 서버 시작 시, 모든 모델을 미리 로드해서 딕셔너리에 저장 (전역)
MODEL_CACHE = {}

def preload_models(model_paths, device):
    global MODEL_CACHE
    if not MODEL_CACHE:  # 한 번만 로딩
        for idx, path in enumerate(model_paths):
            model = load_model(path, device)
            model.eval()
            MODEL_CACHE[idx] = model
    return MODEL_CACHE

def disease_inference_sequential(image, model_paths, preprocess_funcs, disease_names, device):
    severity_labels = ["정상", "경증", "중증"]
    results = []
    raw_preds = []

    # 모델 사전 로드
    models = preload_models(model_paths, device)

    for idx, (preprocess, disease) in enumerate(zip(preprocess_funcs, disease_names)):
        model = models[idx]  # 이미 로드된 모델만 사용!
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class]) * 100

        raw_preds.append(pred_class)

        severity = severity_labels[pred_class] if 0 <= pred_class < len(severity_labels) else "분류불가"

        result = {
            "disease": disease,
            "severity": severity,
            "confidence": f"{confidence:.2f}%"
        }

        if pred_class == 0:
            result["comment"] = "정상 범위입니다. 두피 상태가 양호합니다."
        elif pred_class == 1:
            result["recommendations"] = get_recommendations_by_disease(disease)
        elif pred_class == 2:
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."

        results.append(result)

    # 추론 후 메모리 정리
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        "raw_predictions": raw_preds,
        "results": results
    }
