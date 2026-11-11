"""Label table for Harvard-FairVLMed (run on Hamilton, where the data are).
Source: https://huggingface.co/datasets/harvardairobotics/FairVLMed (downloaded 11 Nov 2026, dataset.zip sha256
01bbfba4...7153), unzipped to /nobackup/tggq61/data/FairVLMed/{Training,Validation,Test}/data_#####.npz.
One npz per patient: slo_fundus (664 x 512 grayscale SLO image), glaucoma (0/1), age, gender, race, ethnicity,
language, maritalstatus. Parts follow the dataset's own split: 7,000 training, 1,000 validation, 2,000 test.
Licence CC BY-NC-ND 4.0: this table stays on Hamilton (not on GitHub).
python make_labels.py  ->  fairvlmed_labels.csv"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/nobackup/tggq61/data/FairVLMed')
PART = {'Training': 'train', 'Validation': 'val', 'Test': 'test'}
rows = []
for folder, part in PART.items():
    for f in sorted((ROOT / folder).glob('data_*.npz')):
        z = np.load(f)
        rows.append({'image_id': f.stem, 'path': str(f), 'part': part, 'label': int(z['glaucoma']),
                     'age': float(z['age']), 'gender': int(z['gender']), 'race': int(z['race']),
                     'ethnicity': int(z['ethnicity']), 'language': int(z['language'])})
df = pd.DataFrame(rows)
df.to_csv('fairvlmed_labels.csv', index=False)
print(df.groupby('part')['label'].agg(['size', 'sum', 'mean']).rename(columns={'size': 'images', 'sum': 'glaucoma'}))
print('race (0 Asian, 1 Black, 2 White):', df['race'].value_counts().sort_index().to_dict(),
      '| gender (0 F, 1 M):', df['gender'].value_counts().sort_index().to_dict())
