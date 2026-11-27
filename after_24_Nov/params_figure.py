"""Trained weights per model (answers the 24 Nov question on the 2-block count). RETFound-Green block (ViT-S, 384 wide):
qkv 443,520 + proj 147,840 + MLP 591,360 + 590,208 + 2 norms 1,536 + 2 layer scales 768 = 1,775,232 weights."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = [('Whole RETFound-Green encoder', 21_832_704, '#9a9a9a'), ('Old CNN from scratch (5 blocks, 32→512)', 4_356_897, '#2f6b9a'),
        ('LoRA + 2 open blocks', 3_662_209, '#c0583a'), ('2 open blocks (blocks 11–12 + final norm + head)', 3_551_617, '#c0583a'),
        ('LoRA in all 12 blocks (+ head)', 110_977, '#c0583a'), ('Disc-mask head (extra, training only)', 385, '#c0583a'),
        ('Frozen features + logistic regression', 385, '#9a9a9a'), ('Smallest new CNN tested (1 block, 8 ch.)', 4_105, '#2f6b9a')]
fig, ax = plt.subplots(figsize=(10, 4.4))
for i, (n, v, c) in enumerate(rows[::-1]):
    ax.barh(i, v, color=c, height=0.62); ax.text(v * 1.15, i, f'{v:,}', va='center', fontsize=9.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9.5); ax.set_xscale('log'); ax.set_xlim(100, 2e8)
ax.set_xlabel('trained weights (log scale)'); ax.spines[['top', 'right']].set_visible(False)
ax.set_title('How many weights each model trains (counted from the models).\nOne RETFound-Green block = 1,775,232 weights, so 2 open blocks = 3.55 M of the 21.8 M', fontsize=10.5, loc='left')
fig.tight_layout(); fig.savefig('figures/trained_weights.png', dpi=170); print('saved')
