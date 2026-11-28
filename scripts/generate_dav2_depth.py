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
    scenes = [args.scene] if args.scene is not None else sorted(
        p.parent for p in (ROOT / "data/llff").glob("*/images")
    )
    images = [
        path for scene in scenes
        for path in sorted((scene / "images").iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vitl", 
        features=256, 
        out_channels=[256, 512, 1024, 1024]
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for index, path in enumerate(images, 1):
        output = path.parent.parent / "depth-maps-DAV2"
        output.mkdir(exist_ok=True)
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        depth = model.infer_image(image, args.input_size)
        depth = torch.nn.functional.interpolate(
            torch.from_numpy(depth)[None, None],
            size=(h // args.downsample, w // args.downsample),
            mode="bicubic", align_corners=False,
        )[0, 0].numpy()
        # Match dpt/utils_io.py: per-image min/max scaling to a 16-bit PNG.
        span = float(depth.max() - depth.min())
        depth = (depth - depth.min()) / span if span > np.finfo(float).eps else np.zeros_like(depth)
        destination = output / f"depth_{path.name.split('.')[0]}.png"
        cv2.imwrite(str(destination), (depth * 65535).astype(np.uint16))
        print(f"[{index}/{len(images)}] {destination}", flush=True)


if __name__ == "__main__":
    main()
