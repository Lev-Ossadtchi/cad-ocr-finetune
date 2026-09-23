#!/usr/bin/env python3
"""Скачать шрифт. В репозитории его нет намеренно: osifont под GPLv3, и
класть его сюда значило бы тянуть лицензию на весь код. Картинки, нарисованные
шрифтом, под лицензию не подпадают — на это есть font exception."""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/hikikomori82/osifont/master/osifont.ttf"
OUT = Path(__file__).parent / "assets" / "osifont.ttf"

OUT.parent.mkdir(exist_ok=True)
if OUT.exists():
    print("шрифт уже на месте:", OUT)
else:
    urllib.request.urlretrieve(URL, OUT)
    print(f"скачан {OUT} ({OUT.stat().st_size} байт)")
