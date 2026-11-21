"""Generate LLFF DAV2 maps using the same storage convention as the MiDaS maps."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "depth-anything-v2"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, nargs="?", help="LLFF scene directory; omit to process all scenes in data/llff")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "submodules/depth-anything-v2/checkpoints/depth_anything_v2_vitl.pth")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--downsample", type=int, default=8)
    args = parser.parse_args()
    if args.downsample < 1 or args.input_size < 1:
        parser.error("input-size and downsample must be positive")
    scenes = [args.scene] if args.scene is not None else sorted(
        p.parent for p in (ROOT / "data/llff").glob("*/images") if p.is_dir()
    )
    images = []
    for scene in scenes:
        image_dir = scene / "images"
        if not image_dir.is_dir():
            parser.error(f"Image directory not found: {image_dir}")
        scene_images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not scene_images:
            parser.error(f"No images found in {image_dir}")
        images.extend(scene_images)
    if not images:
        parser.error(f"No LLFF scenes found in {ROOT / 'data/llff'}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for index, path in enumerate(images, 1):
        output = path.parent.parent / "depth-maps-DAV2"
        output.mkdir(exist_ok=True)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read {path}")
        h, w = image.shape[:2]
        if min(h, w) < args.downsample:
            raise ValueError(f"Image too small for downsample={args.downsample}: {path}")
        depth = model.infer_image(image, args.input_size)
        depth = torch.nn.functional.interpolate(
            torch.from_numpy(depth)[None, None],
            size=(h // args.downsample, w // args.downsample),
            mode="bicubic", align_corners=False,
        )[0, 0].numpy()
        if not np.isfinite(depth).all():
            raise ValueError(f"Non-finite depth prediction: {path}")
        # Match dpt/utils_io.py: per-image min/max scaling to a 16-bit PNG.
        span = float(depth.max() - depth.min())
        depth = (depth - depth.min()) / span if span > np.finfo(float).eps else np.zeros_like(depth)
        destination = output / f"depth_{path.name.split('.')[0]}.png"
        if not cv2.imwrite(str(destination), (depth * 65535).astype(np.uint16)):
            raise OSError(f"Could not write {destination}")
        print(f"[{index}/{len(images)}] {destination}", flush=True)


if __name__ == "__main__":
    main()
