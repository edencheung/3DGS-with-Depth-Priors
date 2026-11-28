"""Triangulate the LLFF training views with COLMAP and their supplied camera poses."""
import argparse
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np

# Use the lightweight camera reader without importing the CUDA training package.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scene"))
from colmap_loader import read_extrinsics_binary, read_intrinsics_binary


def triangulate(scene, output, colmap, n_sparse):
    sparse = scene / "sparse/0"
    cameras = read_intrinsics_binary(str(sparse / "cameras.bin"))
    images = read_extrinsics_binary(str(sparse / "images.bin"))
    ordered = sorted(images.values(), key=lambda image: Path(image.name).stem)
    candidates = [image for i, image in enumerate(ordered) if i % 8 != 0]
    selected = [candidates[round(i)] for i in np.linspace(0, len(candidates) - 1, n_sparse)]
    views = {view.name: view for view in selected}
    camera_ids = {view.camera_id for view in selected}
    camera = cameras[next(iter(camera_ids))]

    known, result = output / "known_poses", output / "triangulated"
    output.mkdir(parents=True)
    known.mkdir()
    result.mkdir()
    listing, database = output / "train_images.txt", output / "database.db"
    listing.write_text("\n".join(views) + "\n", encoding="utf-8")

    def run(*args):
        return subprocess.run([str(colmap), *map(str, args)], check=True)

    params = ",".join(map(str, camera.params))
    # Idenfities features like corners and special textures from each image
    run("feature_extractor", 
        "--database_path", database, 
        "--image_path", scene / "images",
        "--image_list_path", listing, 
        "--ImageReader.single_camera", 1,
        "--ImageReader.camera_model", camera.model,
        "--ImageReader.camera_params", params,
        "--FeatureExtraction.use_gpu", 0, 
        "--FeatureExtraction.num_threads", 4,
        "--FeatureExtraction.max_image_size", 1600)

    with sqlite3.connect(database) as db:
        registered = db.execute(
            "SELECT image_id, name, camera_id FROM images"
        ).fetchall()
    db_camera_id = registered[0][2] # take ID of first camera, all should have the same ID
    calibration = " ".join(map(str, camera.params))
    (known / "cameras.txt").write_text(
        f"{db_camera_id} {camera.model} {camera.width} {camera.height} {calibration}\n",
        encoding="utf-8",
    )
    with (known / "images.txt").open("w", encoding="utf-8") as stream:
        for image_id, name, camera_id in registered:
            view = views[name]
            pose = " ".join(map(str, (*view.qvec, *view.tvec)))
            stream.write(f"{image_id} {pose} {camera_id} {name}\n\n")
    (known / "points3D.txt").touch()

    # matches the special features across images
    run("exhaustive_matcher", 
        "--database_path", database,
        "--FeatureMatching.use_gpu", 0, 
        "--FeatureMatching.num_threads", 4)
    # calculates the 3d position of each matched feature
    run("point_triangulator", 
        "--database_path", database, 
        "--image_path", scene / "images",
        "--input_path", known, 
        "--output_path", result, 
        "--refine_intrinsics", 0,
        "--Mapper.fix_existing_frames", 1, 
        "--Mapper.tri_ignore_two_view_tracks", 0)

    reconstructed = read_extrinsics_binary(str(result / "images.bin"))
    fitted_cameras = read_intrinsics_binary(str(result / "cameras.bin"))
    tracked_points = set()
    for view in reconstructed.values():
        point_ids = view.point3D_ids[view.point3D_ids >= 0]
        tracked_points.update(point_ids)

    run("model_converter", 
        "--input_path", result, 
        "--output_path", result, 
        "--output_type", "TXT")
    print(f"{len(tracked_points)} sparse points from {len(views)} fixed training views in {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="LLFF scene with images and sparse/0")
    parser.add_argument("output", type=Path, help="New directory for the sparse reconstruction")
    parser.add_argument("--colmap", default="colmap", help="Path to COLMAP executable")
    parser.add_argument("--n-sparse", type=int, default=3)
    args = parser.parse_args()
    colmap = shutil.which(args.colmap)
    if colmap is None:
        parser.error(f"COLMAP executable not found: {args.colmap}")
    triangulate(args.scene.resolve(), args.output.resolve(), colmap, args.n_sparse)


if __name__ == "__main__":
    main()

