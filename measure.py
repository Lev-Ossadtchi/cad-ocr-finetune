#!/usr/bin/env python3
"""Замер: сколько подписей читает распознаватель на каждом уровне порчи.

Две цифры на уровень. По символам — доля совпавших знаков (1 − CER), она
показывает, «почти прочитал» или «не понял вовсе». Целиком — доля строк,
совпавших до знака; именно она решает, годится ли строка в сверку без человека.

Обе считаются на тестовых шаблонах, которых не было в обучении.

    python measure.py [сколько строк] [путь к дообученным весам]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from render import LEVELS, render          # noqa: E402
from vocab import sample                   # noqa: E402


def cer(a, b):
    """Расстояние Левенштейна, нормированное на длину эталона."""
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    weights = sys.argv[2] if len(sys.argv) > 2 else None
    import recognizer as R
    model, conv = R.build(weights)
    rows = sample("test", n, seed=7)
    print(f"{'уровень':8} {'по символам':>12} {'целиком':>9}   что это")
    what = {0: "вектор, чистая вырезка", 1: "хороший растр",
            2: "обычный скан", 3: "плохой скан"}
    for lvl in sorted(LEVELS):
        imgs = [render(t, level=lvl, seed=i) for i, (_, t) in enumerate(rows)]
        got = R.read(model, conv, imgs)
        char = 1 - sum(cer(t, g) for (_, t), g in zip(rows, got)) / len(rows)
        exact = sum((t == g) for (_, t), g in zip(rows, got)) / len(rows)
        print(f"{lvl:<8} {char*100:11.1f} % {exact*100:8.1f} %   {what[lvl]}")
    print(f"\nстрок в тесте {len(rows)}, шаблоны только тестовые")
    if "--errors" in sys.argv:
        print("\nчто именно не сошлось (уровень 2):")
        imgs = [render(t, level=2, seed=i) for i, (_, t) in enumerate(rows)]
        got = R.read(model, conv, imgs)
        shown = 0
        for (_, t), g in zip(rows, got):
            if t != g and shown < 8:
                shown += 1
                print(f"   эталон: {t}")
                print(f"   модель: {g}")


if __name__ == "__main__":
    main()
