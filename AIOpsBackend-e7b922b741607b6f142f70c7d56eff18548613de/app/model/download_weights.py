# app/model/download_weights.py

import os
import requests

MODEL_INFOS = [
    (
        "biddem_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v2.0/biddem_compressed.pt"
    ),
    (
        "mise_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v2.0/mise_compressed.pt"
    ),
    (
        "mono_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v2.0/mono_compressed.pt"
    ),
    (
        "mosa_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v2.0/mosa_compressed.pt"
    ),
    (
        "pizi_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v2.0/pizi_compressed.pt"
    ),
    (
        "talmo_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v2.0/talmo_compressed.pt"
    ),
]

os.makedirs("app/model_weight", exist_ok=True)

for fname, url in MODEL_INFOS:
    fpath = os.path.join("app/model_weight", fname)
    if not os.path.exists(fpath):
        print(f"Downloading: {fname}")
        r = requests.get(url)
        if r.status_code == 200:
            with open(fpath, "wb") as f:
                f.write(r.content)
            print(f"Saved: {fpath}")
        else:
            print(f"Failed to download {fname} ({url}) - Status code: {r.status_code}")
    else:
        print(f"Already exists: {fname}")
