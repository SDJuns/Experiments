# app/inference.py

import torch
from app.recommendation.utils import get_recommendations_by_disease
from app.preprocess import disease_keys, disease_names, preprocess_funcs


def disease_inference_sequential(
    image,
    loaded_models: dict,
    preprocess_funcs: list,
    disease_keys: list,
    device: torch.device
):
    """
    이미지와 사전 로드된 모델 사전(loaded_models), 전처리 함수 리스트,
    질환 키 리스트(disease_keys), 장치를 받아 순차적으로 추론을 수행.
    """
    severity_labels = ["정상", "경증", "중증"]
    results = []
    raw_preds = []

    # 질환 키 순회
    for idx, disease_key in enumerate(disease_keys):
        model = loaded_models[disease_key]
        preprocess = preprocess_funcs[idx]

        # 이미지 전처리 및 텐서 변환
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class]) * 100

        raw_preds.append(pred_class)

        # 출력
        severity = severity_labels[pred_class] if 0 <= pred_class < len(severity_labels) else "분류불가"
        display_name = disease_names[idx]
        result = {
            "disease": display_name,
            "severity": severity,
            "confidence": f"{confidence:.2f}%"
        }

        # 결과별 추가 정보
        if pred_class == 0:
            result["comment"] = "정상 범위입니다. 두피 상태가 양호합니다."
        elif pred_class == 1:
            result["recommendations"] = get_recommendations_by_disease(disease_key)
        elif pred_class == 2:
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."

        results.append(result)

    # 메모리 정리
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        "raw_predictions": raw_preds,
        "results": results
    }