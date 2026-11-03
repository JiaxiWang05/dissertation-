"""Download the public comparison-model weights from the Hugging Face hub at pinned revisions, with sha256.
Run on the Hamilton LOGIN node (compute nodes have no internet). Gated models (RETFound, DINOv3) are not handled here:
they need the user's own Hugging Face login.

Usage:
    python download_weights.py --cache /nobackup/tggq61/hf_cache --manifest /nobackup/tggq61/weights/public_manifest.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

PUBLIC = {  # name: (hub repo, revision)
    'dinov2_vitl14': ('timm/vit_large_patch14_dinov2.lvd142m', '4741e1cafbf45415e77074bb0cb42dba76c8684a'),
    'resnet50': ('timm/resnet50.tv_in1k', '78f3ecfdb38e06d9b8397f662e7ab8fee96026fa'),
    'swinv2_tiny': ('timm/swinv2_tiny_window8_256.ms_in1k', '9fc48f948b7a735e67e0a06a9fd5a57d9f5b26e2'),
}
FILE = 'model.safetensors'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='HF_HOME for the downloads')
    ap.add_argument('--manifest', required=True)
    args = ap.parse_args()
    os.environ['HF_HOME'] = args.cache
    from huggingface_hub import hf_hub_download
    records = {}
    for name, (repo, rev) in PUBLIC.items():
        path = Path(hf_hub_download(repo, FILE, revision=rev))
        records[name] = {'repo': repo, 'revision': rev, 'file': str(path), 'bytes': path.stat().st_size,
                         'sha256': sha256(path)}
        print(name, records[name]['bytes'], records[name]['sha256'], flush=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
