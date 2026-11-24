import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
from pathlib import Path
D = Path('/nobackup/tggq61/data/Chaksu')
lab = pd.read_csv('../pipeline/chaksu_labels.csv')
multi, ex, rot = [], [], 0
for q in lab.itertuples():
    f = D / q.image_id.split('/')[0] / '5.0_OD_OC_Mean_Median_Majority_STAPLE' / q.camera / 'Disc' / 'Majority' / f'{q.stem}.tif'
    m = np.asarray(Image.open(f).convert('L')) > 127
    s = np.bincount(ndimage.label(m)[0].ravel())[1:]
    if len(s) > 1:
        multi.append(q.camera); ex.append(1 - s.max() / s.sum())
    rot += Image.open(q.path).getexif().get(274, 1) != 1
print('masks with more than one region:', len(multi), 'of', len(lab), pd.Series(multi).value_counts().to_dict())
if ex:
    print('share of disc pixels outside the largest region: median %.5f, max %.5f' % (np.median(ex), max(ex)))
print('images with EXIF orientation other than 1:', rot)
