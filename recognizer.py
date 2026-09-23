#!/usr/bin/env python3
"""Распознаватель строк: готовый из EasyOCR и он же после дообучения.

Модель — CRNN из EasyOCR (`cyrillic_g2`): свёрточная часть, две LSTM, CTC.
Берём её как есть, чтобы честно сравнивать «до» и «после»: разница в цифрах
должна быть заслугой дообучения, а не другой архитектуры.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_H, IMG_W = 64, 600
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WEIGHTS = Path.home() / ".EasyOCR/model/cyrillic_g2.pth"


def alphabet():
    """Алфавит cyrillic_g2 — берём тот, на котором модель обучалась.

    Свой список писать нельзя: порядок символов задаёт номера классов CTC,
    и при другом порядке готовые веса начнут выдавать не те буквы.
    """
    from easyocr.config import recognition_models
    return recognition_models["gen2"]["cyrillic_g2"]["characters"]


def build(weights=None):
    """Модель и конвертер меток. `weights` — путь к дообученным весам."""
    sys.path.insert(0, str(Path(__file__).parent))
    from easyocr.model.vgg_model import Model
    from easyocr.utils import CTCLabelConverter
    chars = alphabet()
    conv = CTCLabelConverter(chars, {}, {})
    model = Model(num_class=len(conv.character), input_channel=1,
                  output_channel=256, hidden_size=256)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict({k[7:]: v for k, v in sd.items()})
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model, conv


def prep(img):
    w = max(8, min(IMG_W, int(img.width * IMG_H / img.height)))
    im = img.convert("L").resize((w, IMG_H), Image.LANCZOS)
    canvas = Image.new("L", (IMG_W, IMG_H), 255)
    canvas.paste(im, (0, 0))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy((a - 0.5) / 0.5).unsqueeze(0)


def decode(pred, conv):
    idx = pred.max(1)[1].cpu().numpy()
    out, prev = [], -1
    for i in idx:
        i = int(i)
        if i != prev and i != 0:
            out.append(conv.character[i])
        prev = i
    return "".join(out)


def read(model, conv, images, batch=16):
    model.eval().to(DEVICE)
    res = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.stack([prep(im) for im in images[i:i + batch]]).to(DEVICE)
            preds = model(x, None)
            res += [decode(p, conv) for p in preds]
    return res
