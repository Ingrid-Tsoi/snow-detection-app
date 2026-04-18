from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import rasterio
from model.unet import UNet
from utils.inference import predict_mask
from utils.preprocess import crop_tiles
from utils.postprocess import stitch_tiles
import io
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Load Model ==========
MODEL_PATH = "model/snow_val_max_miou.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(n_channels=6, n_classes=2).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state["state_dict"])
model.eval()

# ========== API ==========
@app.get("/")
def home():
    return {"message": "FastAPI is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # 用 rasterio 讀 GeoTIFF
    with rasterio.MemoryFile(contents) as memfile:
        with memfile.open() as src:
            img = src.read().astype(np.float32)
            img = np.nan_to_num(img)
            img = np.clip(img, 0, 65535) / 65535.0

    # crop
    tiles, positions = crop_tiles(img)

    # inference
    pred_tiles = []
    for tile in tiles:
        pred = predict_mask(model, tile, DEVICE)  # (H, W)
        pred_tiles.append(pred.astype(np.float32))

    # stitch
    mask = stitch_tiles(pred_tiles, positions, img.shape)        



    # 轉 PNG
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")




