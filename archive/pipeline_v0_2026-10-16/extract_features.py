"""Extract frozen RETFound-Green features for every image in a labels CSV.

Follows the official loading/preprocessing from
https://github.com/justinengelmann/RETFound_Green (timm vit_small_patch14_reg4_dinov2,
392x392 input, mean/std 0.5, average-pooled tokens -> 384-d feature).

Usage:
    python extract_features.py --labels papila_labels.csv --weights retfoundgreen_statedict.pth \
        --out features/papila_rfg.npz
"""
import argparse
import time

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as T

IMG_SIZE = 392


class ImageCSV(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = read_image(self.paths[i], mode=ImageReadMode.RGB)
        return self.transform(img), i


def load_retfound_green(weights):
    model = timm.create_model('vit_small_patch14_reg4_dinov2', img_size=(IMG_SIZE, IMG_SIZE),
                              num_classes=0, checkpoint_path=weights).eval()
    model.global_pool = 'avg'
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='CSV with columns image_id and path')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(max(1, args.workers))
    print(f'{len(df)} images, device={device}, torch={torch.__version__}, timm={timm.__version__}', flush=True)

    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
                           T.ToDtype(torch.float32, scale=True),
                           T.Normalize([0.5] * 3, [0.5] * 3)])
    loader = DataLoader(ImageCSV(df['path'], transform), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)
    model = load_retfound_green(args.weights).to(device)

    feats = np.zeros((len(df), model.num_features), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for b, (x, idx) in enumerate(loader):
            feats[idx.numpy()] = model(x.to(device)).float().cpu().numpy()
            if b % 5 == 0:
                print(f'  batch {b + 1}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

    np.savez(args.out, features=feats, image_id=np.array(df['image_id'].tolist(), dtype=str))
    print(f'saved {feats.shape} -> {args.out} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
