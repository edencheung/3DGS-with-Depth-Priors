"""
Convert gaussian PLY files to be viewable at https://superspl.at/editor
Note that the PLY file in the original output only contains a fixed RGB value for each gaussian, so the viewer won't be able to visualise colors from different directions.
"""

import argparse
from pathlib import Path

import math
import numpy as np
from plyfile import PlyData, PlyElement


def convert(source, destination):
    ply = PlyData.read(str(source))
    vertices = ply["vertex"].data
    identical_fields = ["x", "y", "z", "opacity"]
    identical_fields += [f"scale_{i}" for i in range(3)]
    identical_fields += [f"rot_{i}" for i in range(4)]

    output_fields = ["x", "y", "z", "nx", "ny", "nz"]
    output_fields += [f"f_dc_{i}" for i in range(3)]
    output_fields += identical_fields[3:]
    result = np.zeros(len(vertices), dtype=[(field, "<f4") for field in output_fields]) # little endian, 4 byte floating points
    for field in identical_fields + ["nx", "ny", "nz"]:
        if field in vertices.dtype.names:
            result[field] = vertices[field]
    for i, channel in enumerate(("red", "green", "blue")):
        result[f"f_dc_{i}"] = (
            vertices[channel].astype(np.float32) / 255.0 - 0.5
        ) * math.sqrt(4 * math.pi)

    with destination.open("xb") as output:
        PlyData(
            [PlyElement.describe(result, "vertex")], text=False, byte_order="<"
        ).write(output)
    print(f"Wrote {len(result):,} Gaussians to {destination}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to point_cloud_color.ply")
    parser.add_argument("--output", type=Path, help="Default: point_cloud_supersplat.ply")
    args = parser.parse_args()
    destination = args.output or args.source.with_name("point_cloud_supersplat.ply")
    convert(args.source, destination)


if __name__ == "__main__":
    main()
