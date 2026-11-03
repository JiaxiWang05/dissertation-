"""Extract frozen RETFound-Green features (384-d) for every image in a labels CSV.

Usage:
    python extract_features.py --labels papila_labels.csv --weights retfoundgreen_statedict.pth \
        --out features/papila_rfg_resize.npz [--preproc resize|fov_square] [--data-root ~/data/dissertation]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader

import rfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='CSV with columns image_id and path')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--preproc', choices=['resize', 'fov_square', 'disc_crop'], default='resize',
                    help="'resize' = official (squash to 392x392); 'fov_square' = crop to the retina, pad to square; "
                         "'disc_crop' = square around the optic disc (needs --boxes)")
    ap.add_argument('--boxes', default=None, help='CSV image_id,cx,cy,disc_diameter from disc_boxes.py')
    ap.add_argument('--crop-scale', type=float, default=3.0, help='disc_crop side length in disc diameters')
    ap.add_argument('--data-root', default=None, help='replace /nobackup/tggq61/data in the CSV paths')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    device = rfg.pick_device()
    torch.set_num_threads(max(1, args.workers))
    print(f'{len(df)} images, preproc={args.preproc}, device={device}, torch={torch.__version__}, '
          f'timm={timm.__version__}', flush=True)

    boxes = None
    if args.preproc == 'disc_crop':
        b = pd.read_csv(args.boxes).set_index('image_id').loc[df['image_id']]
        boxes = list(b[['cx', 'cy', 'disc_diameter']].itertuples(index=False, name=None))
    ds = rfg.FundusImages(rfg.local_paths(df['path'], args.data_root), rfg.eval_transform(), args.preproc,
                          boxes=boxes, crop_scale=args.crop_scale)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    model = rfg.load_retfound_green(args.weights).eval().to(device)

    feats = np.zeros((len(df), model.num_features), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for b, (x, idx) in enumerate(loader):
            feats[idx.numpy()] = model(x.to(device)).float().cpu().numpy()
            if b % 5 == 0:
                print(f'  batch {b + 1}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

    assert np.isfinite(feats).all(), 'non-finite features'
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, features=feats, image_id=np.array(df['image_id'].tolist(), dtype=str),
             preproc=args.preproc, crop_scale=args.crop_scale, weights=str(args.weights))
    print(f'saved {feats.shape} -> {args.out} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
