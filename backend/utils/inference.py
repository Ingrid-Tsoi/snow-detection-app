import torch
import numpy as np

def predict_mask(model, img, device):
    # img shape: (bands, H, W)
    x = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    return pred