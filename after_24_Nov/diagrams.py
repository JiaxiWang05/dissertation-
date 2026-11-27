"""Block diagrams for Raheleh (24 Nov meeting), drawn with one real Chakshu glaucoma image (fold-1 validation image 20;
everything outside the retina blacked out because the photos carry burned-in patient details).
  figures/mask_head_diagram.png   where the disc comes from (expert mask shipped with the dataset), how it becomes the
                                   28 x 28 training target, the training-only mask head, and the expert disc box / crop
  figures/evidence_steps.png      how an evidence map is computed, step by step, with this image's real numbers
python diagrams.py   (needs examples/anon20.png, examples/ex20_mask.png, ../after_22_Nov/attention_maps.npz)"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
J = 20
photo = np.asarray(Image.open(HERE / 'examples' / f'anon{J}.png'))
mask = np.asarray(Image.open(HERE / 'examples' / f'ex{J}_mask.png')).astype(float) / 255
target = mask.reshape(28, 14, 28, 14).mean((1, 3))
z = np.load(HERE.parent / 'after_22_Nov' / 'attention_maps.npz')
TEXT, MUTED, BLUE, ORANGE, RED, GREEN, GREY = '#1a1a1a', '#666666', '#2f6b9a', '#c0583a', '#b5443b', '#2e8b57', '#9a9a9a'


def card(ax, x, y, w, h, title, body='', fc='white', ec=TEXT, dashed=False, tsize=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec=ec, lw=1.3,
                                ls=(0, (4, 3)) if dashed else '-'))
    ax.text(x + w / 2, y + h - 0.18, title, ha='center', va='top', fontsize=tsize, fontweight='bold', color=TEXT)
    if body:
        ax.text(x + w / 2, y + h - 0.55, body, ha='center', va='top', fontsize=9, color=TEXT, linespacing=1.35)


def arrow(ax, p, q, col=TEXT, ls='-', text=None, toff=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, color=col, lw=1.4, ls=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha='center', fontsize=8.5, color=col)


def img(fig, rect, a, cmap=None, title=None, overlay=None, grid=False, vmin=None, vmax=None):
    ax = fig.add_axes(rect); ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    if overlay is not None:
        ax.contour(overlay, levels=[0.5], colors='#00e000', linewidths=1.5)
    if grid:
        n = a.shape[0]; s = n / 28
        for k in range(0, 29, 4):
            ax.axhline(k * s - 0.5, color='white', lw=0.4, alpha=0.6); ax.axvline(k * s - 0.5, color='white', lw=0.4, alpha=0.6)
    if title:
        ax.set_title(title, fontsize=9.5, color=TEXT, pad=4)
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb')
    return ax


# ------------------------------------------------------------------ 1. mask-head diagram
W, H = 18.0, 10.6
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
ax.text(0.3, H - 0.3, 'Disc-mask head: what goes in, what is learned, and what is used at test time', fontsize=16,
        fontweight='bold', va='top', color=TEXT)
ax.text(0.3, H - 0.85, 'One Chákṣu glaucoma image (fold-1 validation). The optic disc is NOT found from pixel contrast: '
        'Chákṣu ships an expert segmentation of the disc for every photo.', fontsize=11, va='top', color=MUTED)
# lane labels
ax.add_patch(Rectangle((0.2, 5.55), W - 0.4, 3.55, fc='#eef4fa', ec='none'))
ax.add_patch(Rectangle((0.2, 0.25), W - 0.4, 5.1, fc='#fbf1ec', ec='none'))
ax.text(0.35, 8.95, 'USED FOR EVERY IMAGE (training and test)', fontsize=10, fontweight='bold', color=BLUE, va='top')
ax.text(0.35, 5.2, 'TRAINING ONLY (dropped at test time)', fontsize=10, fontweight='bold', color=ORANGE, va='top')
# top lane: image -> encoder -> tokens -> mean -> head -> score
img(fig, [0.03, 0.555, 0.135, 0.23], photo, title='1. Photo, 392 × 392 px')
card(ax, 3.35, 6.0, 2.75, 2.55, '2. RETFound-Green', 'ViT-Small, 12 blocks\npre-trained weights frozen\nLoRA rank 4 in every\nattention layer (trained)', fc='#e3f0e3', ec=GREEN)
img(fig, [0.355, 0.555, 0.135, 0.23], photo, grid=True, title='3. 784 patch tokens (28 × 28)')
ax.text(6.55 + 1.2, 5.75, 'each 14 × 14-px patch → 384 numbers', ha='center', fontsize=8.5, color=MUTED)
card(ax, 9.15, 6.0, 2.6, 2.55, '4. Mean of the\n784 tokens', '\n\n384 numbers\nfor the image', fc='white', ec=BLUE)
card(ax, 12.15, 6.0, 2.6, 2.55, '5. Linear head', '\n\nweights w (384)\n+ bias', fc='white', ec=BLUE)
card(ax, 15.15, 6.0, 2.5, 2.55, '6. Glaucoma\nscore', '\n\nthreshold at 90 %\nsensitivity', fc='#f6e0e0', ec=RED)
for a_, b_ in [((2.98, 7.3), (3.35, 7.3)), ((6.1, 7.3), (6.4, 7.3)), ((8.85, 7.3), (9.15, 7.3)), ((11.75, 7.3), (12.15, 7.3)),
               ((14.75, 7.3), (15.15, 7.3))]:
    arrow(ax, a_, b_, BLUE)
# bottom lane: expert mask -> 28x28 target ; tokens -> mask head -> predicted map ; loss
img(fig, [0.03, 0.08, 0.135, 0.23], photo, overlay=mask, title='a. Expert disc mask (green)\nfrom the dataset folder')
ax.text(1.5, 0.45, '5.0_OD_OC_..._STAPLE/<camera>/\nDisc/Majority/<image>.tif', ha='center', fontsize=7.8, color=MUTED, family='monospace')
img(fig, [0.20, 0.08, 0.135, 0.23], target, cmap='Greens', vmin=0, vmax=1, grid=True, title='b. Target: share of each\npatch inside the disc (28 × 28)')
ax.text(4.6, 0.45, 'average over 14 × 14 px', ha='center', fontsize=8.5, color=MUTED)
card(ax, 6.9, 1.4, 3.0, 2.6, 'c. Mask head', 'one linear layer:\n384 numbers → 1 per patch\n(385 weights)\npredicts map b', fc='white', ec=ORANGE, dashed=True)
card(ax, 10.4, 1.4, 3.3, 2.6, 'd. Mask loss', 'cross-entropy + Dice\nbetween c and b,\n× 0.25, added to the\nglaucoma loss', fc='white', ec=ORANGE, dashed=True)
card(ax, 14.2, 1.4, 3.45, 2.6, 'e. Why it can help', 'gradients from d flow back\ninto the LoRA weights:\nthe tokens must also encode\nwhere the disc is', fc='white', ec=ORANGE, dashed=True)
arrow(ax, (7.95, 5.6), (8.2, 4.0), ORANGE, '--', 'same tokens', (0.9, 0.0))
arrow(ax, (2.95, 1.9), (3.55, 1.9), ORANGE)
arrow(ax, (6.0, 2.7), (6.9, 2.7), ORANGE, '--')
arrow(ax, (9.9, 2.7), (10.4, 2.7), ORANGE, '--')
arrow(ax, (13.7, 2.7), (14.2, 2.7), ORANGE, '--')
ax.text(W - 0.3, 0.35, 'Result (5 folds, 3 seeds): disc Dice 0.93–0.94; healthy images flagged at 90 % sensitivity 21 % vs 39 % without the head; '
        'AUROC +0.04 (CI 0.02–0.06); AUPRC tied.', ha='right', fontsize=9.5, color=TEXT)
fig.savefig(OUT / 'mask_head_diagram.png', dpi=170, facecolor='white'); plt.close(fig)

# ------------------------------------------------------------------ 1b. disc box / crop (the "expert box" input)
lab, n = ndimage.label(mask > 0.5); sizes = np.bincount(lab.ravel()); sizes[0] = 0; reg = lab == sizes.argmax()
ys, xs = np.nonzero(reg); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
cx, cy, d = (x0 + x1) / 2, (y0 + y1) / 2, ((x1 - x0) + (y1 - y0)) / 2; half = 1.5 * d
fig = plt.figure(figsize=(14, 4.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 4.2); ax.axis('off')
ax.text(0.2, 4.05, 'Disc-crop input: the box also comes from the expert mask, not from image intensities', fontsize=13, fontweight='bold', va='top')
a1 = img(fig, [0.02, 0.06, 0.2, 0.72], photo, overlay=mask, title='1. Expert disc mask')
a2 = img(fig, [0.27, 0.06, 0.2, 0.72], photo, title='2. Largest region → bounding box')
a2.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='#00e000', lw=1.5))
a3 = img(fig, [0.52, 0.06, 0.2, 0.72], photo, title='3. Square of 3 disc diameters')
a3.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, ec='yellow', lw=1.5))
crop = photo[int(max(cy - half, 0)):int(cy + half), int(max(cx - half, 0)):int(cx + half)]
img(fig, [0.77, 0.06, 0.2, 0.72], crop, title='4. Crop, resized to 392 px')
for xa in (3.05, 6.55, 10.05):
    arrow(ax, (xa, 1.9), (xa + 0.45, 1.9))
fig.savefig(OUT / 'disc_box_diagram.png', dpi=170, facecolor='white'); plt.close(fig)

# ------------------------------------------------------------------ 2. evidence map, step by step
c = z['LoRA + disc-mask head'][J]; c0 = z['LoRA'][J]; d28 = z['disc28'][J].reshape(28, 28)
lim = np.percentile(np.abs(np.concatenate([c, c0])), 99)
W, H = 18.0, 8.4
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
ax.text(0.3, H - 0.3, 'How an evidence map is computed (exact, no gradients)', fontsize=16, fontweight='bold', va='top')
ax.text(0.3, H - 0.85, 'Because the glaucoma score is a linear head on the MEAN of the 784 patch tokens, it splits exactly into one '
        'number per patch.', fontsize=11, va='top', color=MUTED)
img(fig, [0.02, 0.36, 0.15, 0.4], photo, grid=True, title='1. 784 patches')
card(ax, 3.3, 3.3, 3.2, 3.0, '2. Token per patch', 'patch i → tᵢ\n(384 numbers, after\nthe last block and norm)', ec=GREEN, fc='#e3f0e3')
card(ax, 7.0, 3.3, 3.6, 3.0, '3. Contribution per patch', 'cᵢ = w · tᵢ / 784\n\nw = the 384 head weights\n> 0 pushes to glaucoma\n< 0 pushes to healthy', ec=BLUE)
ax2 = img(fig, [0.615, 0.36, 0.15, 0.4], c.reshape(28, 28), cmap='RdBu_r', vmin=-lim, vmax=lim, overlay=d28,
          title='4. The 784 cᵢ as a map\n(red = glaucoma, blue = healthy)')
card(ax, 14.25, 3.3, 3.45, 3.0, '5. Back to the score', f'logit = bias + Σ cᵢ\nΣ cᵢ = {c.sum():+.2f} for this image\n\nsame number the model\nuses to decide', ec=RED, fc='#f6e0e0')
for a_, b_ in [((3.0, 4.8), (3.3, 4.8)), ((6.5, 4.8), (7.0, 4.8)), ((10.6, 4.8), (11.05, 4.8)), ((13.85, 4.8), (14.25, 4.8))]:
    arrow(ax, a_, b_)
cax = fig.add_axes([0.62, 0.30, 0.14, 0.018]); cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap='RdBu_r'), cax=cax, orientation='horizontal')
cb.set_ticks([-lim, 0, lim]); cb.set_ticklabels([f'{-lim:.3f}', '0', f'{lim:.3f}']); cb.ax.tick_params(labelsize=8)
ax.text(0.3, 1.75, 'Read with the disc (green outline):', fontsize=11, fontweight='bold')
ax.text(0.3, 1.25, '•  Disc contrast = (mean c over disc patches − mean c elsewhere) / s.d. of c. On glaucoma images: LoRA 0.0, LoRA + disc-mask head 2.7 s.d., '
        'so the head makes the disc stand out.', fontsize=10)
ax.text(0.3, 0.8, '•  Share of the total |c| inside the disc stays near the disc’s area (1.7 % vs 2.1 %): the decision still draws on the whole image.', fontsize=10)
ax.text(0.3, 0.35, '•  The CNN uses the same idea on its 8 × 8 grid (class activation map). Map shown: LoRA + disc-mask head; '
        'one shared colour scale; values are contributions to the logit.', fontsize=10, color=MUTED)
fig.savefig(OUT / 'evidence_steps.png', dpi=170, facecolor='white'); plt.close(fig)
print('saved')
