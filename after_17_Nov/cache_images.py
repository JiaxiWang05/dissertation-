"""Cache the images once as uint8 arrays for the image-level models (CNN baseline, unfrozen RETFound-Green).
Chakshu: every image of ../pipeline/chaksu_labels.csv, plain resize (as RETFound-Green's official preprocessing) to
  256 x 256 (CNN) and 392 x 392 (RETFound-Green), RGB.  FairVLMed: the SLO image of every patient, 256 x 256, 1 channel.
Arrays go to /nobackup (FairVLMed images must not leave Hamilton).  python cache_images.py chaksu|fairvlmed"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

OUT = Path('/nobackup/tggq61/retfound_green/cache'); OUT.mkdir(parents=True, exist_ok=True)
ds = sys.argv[1]
if ds == 'chaksu':
    lab = pd.read_csv('../pipeline/chaksu_labels.csv')
    for size in (256, 392):
        arr = np.zeros((len(lab), 3, size, size), np.uint8)
        for i, p in enumerate(lab['path']):
            im = ImageOps.exif_transpose(Image.open(p)).convert('RGB').resize((size, size), Image.BILINEAR)
            arr[i] = np.asarray(im).transpose(2, 0, 1)
        np.save(OUT / f'chaksu_{size}.npy', arr)
        print('chaksu', size, arr.shape, flush=True)
    np.save(OUT / 'chaksu_ids.npy', lab['image_id'].to_numpy().astype(str))
else:
    lab = pd.read_csv('../fairvlmed/fairvlmed_labels.csv')
    arr = np.zeros((len(lab), 1, 256, 256), np.uint8)
    for i, p in enumerate(lab['path']):
        a = np.load(p)['slo_fundus']
        arr[i, 0] = np.asarray(Image.fromarray(a).resize((256, 256), Image.BILINEAR))
    np.save(OUT / 'fairvlmed_256.npy', arr); np.save(OUT / 'fairvlmed_ids.npy', lab['image_id'].to_numpy().astype(str))
    print('fairvlmed', arr.shape)
