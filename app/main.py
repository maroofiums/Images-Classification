from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import PredictionResponse
from src.inference import load_model, predict_image

app = FastAPI(
    title="Image Classification API",
    description="CIFAR-10 Image Classification using PyTorch",
    version="1.0.0",
)


model = load_model()


@app.get("/")
def home():
    return {
        "message": "Image Classification API is running."
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    suffix = Path(file.filename).suffix

    temp_path = Path("temp") / f"{uuid.uuid4()}{suffix}"

    temp_path.parent.mkdir(
        exist_ok=True,
        parents=True,
    )

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    try:

        prediction = predict_image(
            temp_path,
            model,
        )

        return PredictionResponse(
            predicted_class=prediction["class"],
            confidence=prediction["confidence"],
        )

    finally:

        if temp_path.exists():
            temp_path.unlink()