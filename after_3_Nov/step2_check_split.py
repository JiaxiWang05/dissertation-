"""Step 2: check the split (stratified, no eye in two parts) and print the table for the meeting."""
from common import load_data

lab, X, y = load_data()
print('features:', X.shape, '  glaucoma:', y.sum(), '  healthy:', len(y) - y.sum())

table = lab.groupby('part').agg(images=('label_majority', 'size'), glaucoma=('label_majority', 'sum'))
table['healthy'] = table['images'] - table['glaucoma']
table['glaucoma_%'] = (100 * table['glaucoma'] / table['images']).round(1)
print(table.loc[['train', 'val', 'test']])

eyes_in_two_parts = (lab.groupby('eye')['part'].nunique() > 1).sum()
print('eyes in more than one part:', eyes_in_two_parts, '(must be 0)')
print(lab.groupby(['camera', 'part']).size().unstack()[['train', 'val', 'test']])
