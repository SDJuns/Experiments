import app.model.download_weights

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.predict import router as predict_router

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aiopsfrontend.onrender.com",  # 실제 프론트엔드 도메인
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# predict 라우터 등록
app.include_router(predict_router, prefix="/api", tags=["prediction"])
