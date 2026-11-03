"""Synthetic stand-in data to check the code paths end to end. The numbers it produces mean nothing.

Writes, under OUT_DIR: 60 fake PAPILA-style images (+ papila_synth.csv) and random 384-d features for every
row of the real label tables (papila_randfeats.npz, chaksu_randfeats.npz), weakly linked to the labels.
    python tests/make_synthetic.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parents[1]
root = Path(sys.argv[1])
rng = np.random.default_rng(0)
pap = pd.read_csv(HERE / 'papila_labels.csv')
cha = pd.read_csv(HERE / 'chaksu_labels.csv')


def fake_fundus(path, w, h, glaucoma):
    im = Image.new('RGB', (w, h))
    d = ImageDraw.Draw(im)
    r, cx, cy = min(w, h) * 0.45, w / 2, h / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150 + int(rng.integers(-20, 20)), 60, 30))
    cup = 0.35 + 0.25 * glaucoma + rng.normal(0, 0.08)
    dr, ox = r * 0.15, cx + r * 0.4
    d.ellipse([ox - dr, cy - dr, ox + dr, cy + dr], fill=(240, 200, 120))
    d.ellipse([ox - dr * cup, cy - dr * cup, ox + dr * cup, cy + dr * cup], fill=(255, 245, 210))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)


sub = pd.concat([pap[pap.diagnosis == 'healthy'].head(30), pap[pap.diagnosis == 'glaucoma'].head(30)])
for _, r in sub.iterrows():
    fake_fundus(Path(r.path.replace('/nobackup/tggq61/data', str(root))), 516, 387, r.diagnosis == 'glaucoma')
sub.to_csv(root / 'papila_synth.csv', index=False)
for name, df, y in [('papila', pap, pap.diagnosis.eq('glaucoma').to_numpy()),
                    ('chaksu', cha, cha.label_majority.to_numpy())]:
    X = rng.normal(size=(len(df), 384)).astype(np.float32)
    X[:, :5] += 0.8 * y[:, None]
    np.savez(root / f'{name}_randfeats.npz', features=X, image_id=np.array(df.image_id.tolist(), dtype=str))
print('synthetic data in', root)
