from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from .inference import BrainMRIPredictor, HEATMAP_DIR
from .inference_light import BrainMRIPredictorLight

app = FastAPI(
    title="Brain MRI Tumor API",
    description="Eğitim amaçlı beyin MR sınıflandırma + ısı haritası çıktısı. Tıbbi tanı yerine geçmez.",
    version="0.1.0",
)

# Production CORS settings
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://your-domain.com",  # Bu kısmı domain ile değiştir
    "*"  # Geçici olarak tüm originlere izin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

predictor = None
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@app.on_event("startup")
def _load_model():
    global predictor
    try:
        # Memory limiti için light model kullan
        predictor = BrainMRIPredictorLight()
        print(" Light model yüklendi (Memory optimized)")
    except Exception as e:
        print(f" Model yüklenemedi: {e}")
        try:
            # Fallback to original model
            predictor = BrainMRIPredictor()
            print(" Original model yüklendi (Fallback)")
        except Exception as e2:
            print(f" Fallback de başarısız: {e2}")
            predictor = None
            print(f"Model load failed: {e}")


@app.post("/predict")
async def predict_mri(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Sadece JPEG/PNG destekleniyor.")

    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Model yüklenemedi. Önce eğitimi çalıştırıp models/brain_mri_classifier.pth üret.",
        )

    image_bytes = await image.read()
    try:
        result = predictor.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    result["heatmap_url"] = f"/heatmaps/{result['heatmap_filename']}"
    return result


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/heatmaps/{filename}")
def get_heatmap(filename: str):
    path = HEATMAP_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Heatmap bulunamadı.")
    return FileResponse(path)

@app.get("/original/{filename}")
def get_original(filename: str):
    path = HEATMAP_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Orijinal görüntü bulunamadı.")
    return FileResponse(path)

