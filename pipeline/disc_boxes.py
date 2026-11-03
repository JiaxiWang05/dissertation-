"""Optic-disc centre and diameter for every image, from the datasets' own expert annotations.

PAPILA: the two experts' disc contours (ExpertsSegmentations/Contours/RET###OD_disc_exp{1,2}.txt), averaged.
Chakshu: the 5-expert majority disc mask (5.0_OD_OC_Mean_Median_Majority_STAPLE/<camera>/Disc/Majority/<stem>.tif).
Diameter = mean of the bounding box's width and height.

Usage (Hamilton):
    python disc_boxes.py --papila-labels papila_labels.csv --chaksu-labels chaksu_labels.csv \
        --out /nobackup/tggq61/data/labels/disc_boxes.csv --qc disc_crops_qc.jpg
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

DATA = Path('/nobackup/tggq61/data')


def papila_box(image_id):
    boxes = []
    for exp in (1, 2):
        f = DATA / 'PAPILA_root/ExpertsSegmentations/Contours' / f'{image_id}_disc_exp{exp}.txt'
        if f.exists():
            xy = np.loadtxt(f)
            boxes.append([xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()])
    x0, x1, y0, y1 = np.mean(boxes, axis=0)
    return (x0 + x1) / 2, (y0 + y1) / 2, ((x1 - x0) + (y1 - y0)) / 2, len(boxes)


def chaksu_box(row):
    f = (DATA / 'Chaksu' / row['split'].title() / '5.0_OD_OC_Mean_Median_Majority_STAPLE' / row['camera'] / 'Disc'
         / 'Majority' / f"{row['stem']}.tif")
    m = np.asarray(Image.open(f).convert('L')) > 127
    lab, n = ndimage.label(m)
    if n > 1:  # keep the largest blob; a few masks carry small stray marks
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ys, xs = np.nonzero(m)
    return (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, ((xs.max() - xs.min()) + (ys.max() - ys.min())) / 2, 1


def qc_montage(rows, out, scale=3.0, tile=260):
    """Full image (squashed, as the paper does) above the disc crop, for a few example images."""
    canvas = Image.new('RGB', (tile * len(rows), tile * 2), 'white')
    for i, r in enumerate(rows):
        im = Image.open(r['path']).convert('RGB')
        canvas.paste(im.resize((tile, tile)), (i * tile, 0))
        half = scale * r['disc_diameter'] / 2
        crop = im.crop((int(r['cx'] - half), int(r['cy'] - half), int(r['cx'] + half), int(r['cy'] + half)))
        canvas.paste(crop.resize((tile, tile)), (i * tile, tile))
    canvas.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papila-labels', required=True)
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--qc', default=None)
    args = ap.parse_args()
    rows = []
    pap = pd.read_csv(args.papila_labels)
    for _, r in pap.iterrows():
        cx, cy, d, n = papila_box(r['image_id'])
        rows.append({'image_id': r['image_id'], 'dataset': 'PAPILA', 'camera': 'PAPILA', 'path': r['path'],
                     'cx': cx, 'cy': cy, 'disc_diameter': d, 'width': r['width'], 'height': r['height'], 'n_annot': n})
    cha = pd.read_csv(args.chaksu_labels)
    for _, r in cha.iterrows():
        cx, cy, d, n = chaksu_box(r)
        rows.append({'image_id': r['image_id'], 'dataset': 'Chaksu', 'camera': r['camera'], 'path': r['path'],
                     'cx': cx, 'cy': cy, 'disc_diameter': d, 'width': r['width'], 'height': r['height'], 'n_annot': n})
    df = pd.DataFrame(rows)
    df['disc_to_width'] = df['disc_diameter'] / df['width']
    df.drop(columns='path').round(2).to_csv(args.out, index=False)
    print(df.groupby('camera')[['disc_diameter', 'disc_to_width']].describe().round(3).to_string())
    if args.qc:
        picks = [df[df['camera'] == c].iloc[3] for c in ['PAPILA', 'Remidio', 'Bosch', 'Forus']]
        qc_montage(picks + [df[df['camera'] == 'PAPILA'].iloc[40]], args.qc)
        print('QC montage:', args.qc)


if __name__ == '__main__':
    main()
