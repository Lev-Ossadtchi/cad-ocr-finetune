#!/usr/bin/env python3
"""Дообучение распознавателя под чертёжный шрифт.

Готовая модель училась на фотографиях вывесок и сканах книг. Чертёжный шрифт
для неё чужой: буквы вертикальные, равномерной толщины, со скруглёнными
концами, а рядом стоят знаки, которых в обычном тексте почти не бывает —
«DN/OD», «SDR17-63х3,8», «i=0,005». Отсюда и ошибки: «/» читается как «I»,
кириллическая «х» как латинская «x».

Учим на синтетике: шрифт открытый (osifont, ГОСТ 2.304), словарь собирается по
шаблонам записи из ГОСТ и ТУ. Тестовые шаблоны в обучение не попадают вовсе —
иначе замер показал бы память, а не чтение.

    python train.py [эпох] [строк на эпоху]
"""
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from recognizer import DEVICE, IMG_H, IMG_W, build, prep   # noqa: E402
from render import LEVELS, render                          # noqa: E402
from vocab import sample                                   # noqa: E402

OUT = Path(__file__).parent / "cad-tuned.pth"


class Synth(Dataset):
    """Подписи рисуются на лету: каждая эпоха видит новый шум и новый кегль."""

    def __init__(self, rows, epoch=0):
        self.rows = rows
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        _, text = self.rows[i]
        seed = i + self.epoch * 100003
        lvl = random.Random(seed).choice(sorted(LEVELS))
        img = render(text, level=lvl, seed=seed)
        return prep(img), text


def collate(batch):
    xs, texts = zip(*batch)
    return torch.stack(xs), list(texts)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze", action="store_true",
                    help="заморозить свёрточную часть: учится только чтение "
                         "последовательности, а признаки остаются прежними")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    epochs, per_epoch = a.epochs, a.rows
    model, conv = build()
    if a.freeze:
        for p_ in model.FeatureExtraction.parameters():
            p_.requires_grad = False
    rows = sample("train", per_epoch, seed=11)
    val = sample("val", 200, seed=12)
    print(f"обучающих подписей {len(rows)}, проверочных {len(val)}, устройство {DEVICE}, "
          f"lr {a.lr}, свёртка {'заморожена' if a.freeze else 'учится'}")

    model.to(DEVICE).train()
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                        lr=a.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    t0 = time.time()
    for ep in range(epochs):
        ds = Synth(rows, epoch=ep)
        dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
        total = seen = 0
        for x, texts in dl:
            x = x.to(DEVICE)
            idx, lens = conv.encode(texts, batch_max_length=IMG_W)
            preds = model(x, None).log_softmax(2).permute(1, 0, 2)
            inp_len = torch.full((x.size(0),), preds.size(0), dtype=torch.long)
            loss = loss_fn(preds, idx.to(DEVICE), inp_len, lens)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * x.size(0); seen += x.size(0)
        torch.save(model.state_dict(), a.out)
        print(f"эпоха {ep+1}/{epochs}: потеря {total/seen:.4f}, "
              f"прошло {time.time()-t0:.0f} с, веса сохранены")
    print("готово:", a.out)


if __name__ == "__main__":
    main()
