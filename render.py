#!/usr/bin/env python3
"""Подпись с чертежа: картинка так, как её видит распознаватель.

Лист чертежа приходит в PDF двумя способами. Либо текст переведён в кривые —
тогда вырезка чистая, буквы тонкие и ровные. Либо лист вставлен картинкой со
сканера или из растрового экспорта — тогда к буквам добавляются размытие,
шум, наклон в доли градуса и потеря тонких штрихов. Обучать и мерить надо на
обоих, иначе цифры получатся про лабораторию, а не про работу.

Портится картинка не «вообще», а лесенкой: у каждого уровня свой набор помех,
и в замере видно, где модель ломается. Уровень 0 — вектор, 3 — плохой скан.
"""
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "assets/osifont.ttf"
H = 64                      # высота вырезки, как её ждёт распознаватель
PAD = 6

LEVELS = {
    0: dict(blur=0.0, noise=0.0, skew=0.0, fade=0.00, jpeg=0),
    1: dict(blur=0.4, noise=4.0, skew=0.3, fade=0.10, jpeg=0),
    2: dict(blur=0.8, noise=9.0, skew=0.6, fade=0.25, jpeg=55),
    3: dict(blur=1.2, noise=14.0, skew=1.0, fade=0.40, jpeg=40),
}


def _jpeg(img, q):
    import io
    b = io.BytesIO()
    img.convert("L").save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("L")


def render(text, level=0, seed=0, font_path=FONT, size=None):
    r = random.Random(seed)
    p = LEVELS[level]
    pt = size or r.randint(28, 44)
    font = ImageFont.truetype(font_path, pt)
    w = int(font.getlength(text)) + 2 * PAD
    img = Image.new("L", (w, pt + 2 * PAD), 255)
    ImageDraw.Draw(img).text((PAD, PAD), text, font=font, fill=0)

    if p["skew"]:
        img = img.rotate(r.uniform(-p["skew"], p["skew"]), expand=True,
                         fillcolor=255, resample=Image.BILINEAR)
    if p["fade"]:
        # Печать и скан «съедают» тонкие штрихи чертёжного шрифта: буквы не
        # рвутся, а бледнеют. Осветляем чернила, фон не трогаем.
        a = np.asarray(img, dtype=np.float32)
        img = Image.fromarray((255 - (255 - a) * (1 - r.uniform(0, p["fade"]))).astype(np.uint8))
    if p["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0, p["blur"])))
    if p["noise"]:
        a = np.asarray(img, dtype=np.float32)
        a += np.random.default_rng(seed).normal(0, p["noise"], a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if p["jpeg"]:
        img = _jpeg(img, p["jpeg"])

    scale = H / img.height
    return img.resize((max(8, int(img.width * scale)), H), Image.LANCZOS)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path("samples")
    out.mkdir(exist_ok=True)
    text = sys.argv[1] if len(sys.argv) > 1 else "Труба КОРСИС SN8 DN/OD 160 ТУ 22.21.21-001-73011750-2021"
    for lvl in LEVELS:
        im = render(text, level=lvl, seed=lvl)
        im.save(out / f"level{lvl}.png")
        print(f"уровень {lvl}: {im.width}×{im.height}")
