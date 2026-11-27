"""Export candidate example images for the mask-head block diagram: fold-1 validation glaucoma images that have evidence
maps (../after_22_Nov/attention_maps.npz), as the 392 px photo, its expert disc mask and the 28 x 28 target. The photos
are checked by eye for burned-in patient details before any is used.  python export_examples.py -> examples/"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent; OUT = HERE / 'examples'; OUT.mkdir(exist_ok=True)
CACHE = Path('/nobackup/tggq61/retfound_green/cache')
z = np.load(HERE.parent / 'after_22_Nov' / 'attention_maps.npz')
ids = np.load(CACHE / 'chaksu_ids.npy'); pos = {k: i for i, k in enumerate(ids)}
X = np.load(CACHE / 'chaksu_392.npy', mmap_mode='r'); M = np.load(CACHE / 'chaksu_mask392.npy', mmap_mode='r')
gl = [i for i in range(len(z['y'])) if z['y'][i] == 1][:8]
for j in gl:
    r = pos[z['image_id'][j]]
    Image.fromarray(np.asarray(X[r]).transpose(1, 2, 0)).save(OUT / f'ex{j}_photo.png')
    Image.fromarray(np.asarray(M[r])).save(OUT / f'ex{j}_mask.png')
    print(j, z['image_id'][j])
