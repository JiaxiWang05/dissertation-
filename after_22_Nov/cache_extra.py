"""22 Nov (ideas borrowed from EndoWAM): cache two more uint8 arrays aligned with ../after_17_Nov's chaksu_392.npy.
- chaksu_mask392.npy: the 5-expert majority optic-disc mask (5.0_OD_OC_Mean_Median_Majority_STAPLE/<camera>/Disc/
  Majority/<stem>.tif; Chakshu provides disc masks only, no cup masks), resized like the images (plain bilinear resize
  to 392 x 392), values 0..255 = share of the pixel inside the disc.
- chaksu_disc392.npy: a square crop around the optic disc (side 3 disc diameters, the crop of the October disc-crop
  experiment, box from /nobackup/tggq61/data/labels/disc_boxes.csv = the expert mask), resized to 392 x 392.
Rows follow chaksu_ids.npy (all 1,345 images; the training code only ever loads the 1,141 training + validation rows).
python cache_extra.py   (CPU)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.io import ImageReadMode, read_image

sys.path.insert(0, '../pipeline')
import rfg  # noqa: E402

CACHE = Path('/nobackup/tggq61/retfound_green/cache')
DATA = Path('/nobackup/tggq61/data/Chaksu')
lab = pd.read_csv('../pipeline/chaksu_labels.csv')
ids = np.load(CACHE / 'chaksu_ids.npy')
assert (ids == lab['image_id'].to_numpy().astype(str)).all()
boxes = pd.read_csv('/nobackup/tggq61/data/labels/disc_boxes.csv').query("dataset == 'Chaksu'").set_index('image_id').loc[lab['image_id']]

masks = np.zeros((len(lab), 392, 392), np.uint8)
crops = np.zeros((len(lab), 3, 392, 392), np.uint8)
for i, r in enumerate(lab.itertuples()):
    folder = r.image_id.split('/')[0]                     # Train or Test
    m = Image.open(DATA / folder / '5.0_OD_OC_Mean_Median_Majority_STAPLE' / r.camera / 'Disc' / 'Majority' / f'{r.stem}.tif')
    assert m.size == (r.width, r.height), (r.image_id, m.size)
    m = (np.asarray(m.convert('L'), np.float32) / 255.0)
    masks[i] = np.round(255 * np.asarray(Image.fromarray(m, mode='F').resize((392, 392), Image.BILINEAR))).clip(0, 255)
    img = read_image(r.path, mode=ImageReadMode.RGB)
    b = boxes.loc[r.image_id]
    c = rfg.disc_crop(img, b['cx'], b['cy'], b['disc_diameter'], scale=3.0)
    crops[i] = T.functional.resize(c, [392, 392], antialias=True).numpy()
    if i % 200 == 0:
        print(i, r.image_id, 'disc share', round(masks[i].mean() / 255, 4), flush=True)
print('empty masks:', int((masks.max((1, 2)) == 0).sum()), 'mean disc share', round(masks.mean() / 255, 4))
np.save(CACHE / 'chaksu_mask392.npy', masks); np.save(CACHE / 'chaksu_disc392.npy', crops)
print('saved', masks.shape, crops.shape)
