#!/usr/bin/env python3
"""Веса PyTorch -> ONNX: то, что реально уезжает заказчику.

Распознаватель в PyTorch требует у заказчика полтора гигабайта библиотек и
прав администратора, ONNX — пятнадцать мегабайт модели и двенадцать
библиотеки. Поэтому обучение идёт в torch, а поставка в ONNX, и рядом с
моделью кладётся список символов: без него выход сети — просто числа.

    python export_onnx.py cad-v3.pth model.onnx
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from recognizer import IMG_H, IMG_W, build            # noqa: E402


def export(weights: str, out: str) -> None:
    model, conv = build(weights)
    model.eval()
    dummy = torch.zeros(1, 1, IMG_H, IMG_W)

    class Wrap(torch.nn.Module):
        """У модели easyocr второй аргумент — текст для режима обучения.
        В ONNX он не нужен, и обёртка убирает его из подписи."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image):
            return self.m(image, None)

    # dynamo=False — старый экспортёр. Новый раскладывает веса в отдельный
    # файл .onnx.data рядом с моделью, а заказчику уезжает один файл: два
    # потеряются при первой же пересылке.
    # Пачку картинок модель не получает: вырезки читаются по одной, ячейка за
    # ячейкой. Зато при плавающем размере пачки старый экспортёр спотыкается
    # на адаптивном пуле — «output_size is not constant». Поэтому размер
    # зафиксирован.
    torch.onnx.export(Wrap(model), dummy, out,
                      input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    chars = Path(out).with_suffix(".chars.json")
    chars.write_text(json.dumps(conv.character, ensure_ascii=False), encoding="utf-8")
    size = Path(out).stat().st_size / 1e6
    print(f"{out}: {size:.1f} МБ, символов {len(conv.character)}, список рядом — {chars.name}")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "cad-v3.pth"
    o = sys.argv[2] if len(sys.argv) > 2 else "model.onnx"
    export(w, o)
