"""Export Apple Depth Pro 2 inverse-depth maps for LLFF training.

Run in the Depth Pro 2 environment, not dngaussian-modern.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules/ml-depth-pro-2/src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, nargs="?", help="LLFF scene directory; omit for all scenes in data/llff")
    parser.add_argument("--model", default="depthpro2-l", choices=["depthpro2-s", "depthpro2-b", "depthpro2-l", "depthpro2-h"])
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--n_sparse", type=int, default=3, help="Training views per scene; match train_llff.py --n_sparse")
    args = parser.parse_args()
    if args.downsample < 1:
        parser.error("downsample must be positive")
    if args.n_sparse < 1:
        parser.error("n_sparse must be positive")
    # Import the lightweight COLMAP reader without importing scene's CUDA model.
    sys.path.insert(0, str(ROOT / "scene"))
    from colmap_loader import read_extrinsics_binary, read_extrinsics_text
    scenes = [args.scene] if args.scene is not None else sorted(
        p.parent for p in (ROOT / "data/llff").glob("*/images") if p.is_dir()
    )
    images = []
    for scene in scenes:
        image_dir = scene / "images"
        if not image_dir.is_dir():
            parser.error(f"Image directory not found: {image_dir}")
        sparse = scene / "sparse/0"
        if (sparse / "images.bin").is_file():
            cameras = read_extrinsics_binary(str(sparse / "images.bin"))
        else:
            cameras = read_extrinsics_text(str(sparse / "images.txt"))
        # Match readColmapSceneInfo: sort registered views, hold out every eighth,
        # then choose evenly spaced views from the remaining candidates.
        registered = sorted(cameras.values(), key=lambda c: Path(c.name).name.split('.')[0])
        candidates = [image_dir / Path(c.name).name for i, c in enumerate(registered) if i % 8 != 0]
        if len(candidates) < args.n_sparse:
            parser.error(f"Not enough training candidates in {scene} for n_sparse={args.n_sparse}")
        selected = {round(i) for i in np.linspace(0, len(candidates) - 1, args.n_sparse)}
        scene_images = [p for i, p in enumerate(candidates) if i in selected]
        for path in scene_images:
            if not path.is_file():
                parser.error(f"Training image not found: {path}")
        print(f"{scene.name}: selected {', '.join(p.name for p in scene_images)}", flush=True)
        images.extend(scene_images)
    if not images:
        parser.error(f"No LLFF scenes found in {ROOT / 'data/llff'}")

    from depth_pro_2 import create_model_and_transforms, load_rgb

    model, transform = create_model_and_transforms(args.model)
    for index, path in enumerate(images, 1):
        image, f_px = load_rgb(path)
        h, w = image.shape[:2]
        if min(h, w) < args.downsample:
            raise ValueError(f"Image too small for downsample={args.downsample}: {path}")
        # Use inverse depth directly: white near, black far, matching MiDaS/DAV2.
        with torch.no_grad():
            depth = model.infer(transform(image), f_px=f_px)["inverse_depth"]
        depth = torch.nn.functional.interpolate(
            depth.float()[None, None],
            size=(h // args.downsample, w // args.downsample),
            mode="bicubic", align_corners=False,
        )[0, 0].cpu().numpy()
        if not np.isfinite(depth).all():
            raise ValueError(f"Non-finite inverse-depth prediction: {path}")
        # Same per-image min/max scaling and 16-bit storage as MiDaS/DAV2.
        span = float(depth.max() - depth.min())
        depth = (depth - depth.min()) / span if span > np.finfo(float).eps else np.zeros_like(depth)
        output = path.parent.parent / "depth-maps-DepthPro"
        output.mkdir(exist_ok=True)
        destination = output / f"depth_{path.name.split('.')[0]}.png"
        Image.fromarray((depth * 65535).astype(np.uint16)).save(destination)
        print(f"[{index}/{len(images)}] {destination}", flush=True)


if __name__ == "__main__":
    main()
