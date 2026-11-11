"""Frozen RETFound-Green features (384 per image) for Harvard-FairVLMed. Run on a Hamilton GPU.
The SLO images are single-channel and dark (median 99th percentile 64 of 255), so two inputs are made:
  raw:     the grayscale image copied into the 3 colour channels, then the official preprocessing
           (resize to 392 x 392, scale to 0-1, normalise with mean and std 0.5)
  stretch: the same after a per-image contrast stretch (1st-99th percentile -> 0-255)
Cross-validation on training + validation decides which one to use; the test part is not looked at.
python extract_features.py --preproc raw|stretch  ->  /nobackup/tggq61/retfound_green/features/fairvlmed_rfg_<preproc>.npz"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '../pipeline')
import rfg  # noqa: E402  (model loading and the official transform)

ap = argparse.ArgumentParser()
ap.add_argument('--preproc', choices=['raw', 'stretch'], required=True)
ap.add_argument('--weights', default='/nobackup/tggq61/weights/retfoundgreen_statedict.pth')
ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()
OUT = f'/nobackup/tggq61/retfound_green/features/fairvlmed_rfg_{args.preproc}.npz'


class SLO(Dataset):
    def __init__(self, paths):
        self.paths, self.tf = list(paths), rfg.eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = np.load(self.paths[i])['slo_fundus'].astype(np.float32)
        if args.preproc == 'stretch':
            lo, hi = np.percentile(a, [1, 99])
            a = np.clip((a - lo) / max(hi - lo, 1.0), 0, 1) * 255
        img = torch.from_numpy(a.round().astype(np.uint8))[None].repeat(3, 1, 1)   # 3 x H x W, uint8
        return self.tf(img), i


df = pd.read_csv('fairvlmed_labels.csv')
device = rfg.pick_device()
model = rfg.load_retfound_green(args.weights).eval().to(device)
loader = DataLoader(SLO(df['path']), batch_size=64, num_workers=args.workers)
feats = np.zeros((len(df), model.num_features), dtype=np.float32)
t0 = time.time()
with torch.inference_mode():
    for b, (x, idx) in enumerate(loader):
        feats[idx.numpy()] = model(x.to(device)).float().cpu().numpy()
        if b % 20 == 0:
            print(f'batch {b + 1}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)
assert np.isfinite(feats).all()
np.savez(OUT, features=feats, image_id=np.array(df['image_id'].tolist(), dtype=str), preproc=args.preproc)
print(f'{len(df)} images on {device}: saved {feats.shape} -> {OUT} in {time.time() - t0:.0f}s')
