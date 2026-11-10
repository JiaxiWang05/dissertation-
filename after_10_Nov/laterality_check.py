"""Left or right eye (10 Nov): Rahele asked whether the image files say which eye they show (OD = right, OS = left),
because flipping all images to one side might help and laterality could also reveal fellow-eye pairs.
Reads only the file headers: every EXIF tag (main, Exif and GPS directories, maker note as text), any XMP / comment
block, and the text strings in the first 128 KB of each file; plus the file and folder names. No pixels are read.
Run on Hamilton (where the images are):  python laterality_check.py  ->  laterality_check.csv"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image

LABELS = '../pipeline/chaksu_labels.csv'
WORDS = re.compile(rb'(?<![A-Za-z])(OD|OS|OU|Right|Left|RIGHT|LEFT|right eye|left eye|R eye|L eye|Laterality|laterality)(?![A-Za-z])')
lab = pd.read_csv(LABELS)
rows, tag_names = [], Counter()
for r in lab.itertuples(index=False):
    path = Path(r.path)
    found, tags = [], {}
    with Image.open(path) as im:
        exif = im.getexif()
        for ifd in [None, ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo]:
            d = exif if ifd is None else exif.get_ifd(ifd)
            for k, v in d.items():
                name = ExifTags.TAGS.get(k, str(k))
                tags[name] = v
                tag_names[name] += 1
        for key in ('comment', 'xmp', 'XML:com.adobe.xmp'):
            if key in im.info:
                tags[key] = im.info[key]
    for name, v in tags.items():
        text = v if isinstance(v, bytes) else str(v).encode('utf-8', 'ignore')
        for m in WORDS.finditer(text):
            found.append(f'{name}: {m.group(0).decode()}')
    head = path.read_bytes()[:131072]
    for s in re.findall(rb'[ -~]{4,}', head):                       # printable text in the header
        for m in WORDS.finditer(s):
            found.append(f'header text: {s[:60].decode()}')
            break
    name_hit = re.search(r'(?<![A-Za-z])(OD|OS|R|L|right|left)(?![A-Za-z])', f'{path.parent.name}/{path.stem}', re.I)
    rows.append({'image_id': r.image_id, 'camera': r.camera, 'n_tags': len(tags),
                 'make': str(tags.get('Make', '')), 'model': str(tags.get('Model', '')),
                 'name_hit': name_hit.group(0) if name_hit else '', 'found': ' | '.join(sorted(set(found)))})
df = pd.DataFrame(rows)
df.to_csv('laterality_check.csv', index=False)
print('images read:', len(df))
print('EXIF tags present (name: number of images):', dict(tag_names.most_common()))
for cam, g in df.groupby('camera'):
    print(f'{cam}: {len(g)} images; camera make/model: {sorted(set(g["make"] + " " + g["model"]))[:3]}; '
          f'with an OD/OS/left/right word in the header: {(g["found"] != "").sum()}; in the file name: {(g["name_hit"] != "").sum()}')
print('examples of hits:', df.loc[df['found'] != '', ['image_id', 'found']].head(10).to_string(index=False))
other = [p for p in Path(lab['path'].iloc[0]).parents[3].rglob('*') if p.suffix.lower() in ('.csv', '.xlsx', '.xls', '.txt', '.json')]
print('other table / text files in the dataset folder:', [str(p) for p in other][:30])
