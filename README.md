# Docker setup instructions

See [README.upstream.md](README.upstream.md) for the original project documentation.

## Build

```bash
docker compose build
```

The PyTorch base image uses Ubuntu 22.04, PyTorch 2.7.1, and CUDA 12.8.
The default build targets RTX 50-series GPUs (`12.0`). For common RTX 30-, 40-,
and 50-series targets:

```bash
CUDA_ARCH_LIST="8.6;8.9;12.0" docker compose build
```

## Dataset

Download the [LLFF dataset](https://drive.google.com/file/d/11PhkBXZZNYTD2emdG1awALlhCnkq7aN-/view?usp=drive_link) and put it under `data/llff/`

## Generate depth maps

### MiDaS DPT Hybrid (default)

```bash
docker compose run --rm dngaussian \
  python dpt/get_depth_map_for_llff_dtu.py --root_path data/llff --benchmark LLFF
```

Output: `data/llff/<scene>/depth_maps/`.

### Depth Anything V2 Large

Place `depth_anything_v2_vitl.pth` in
`submodules/depth-anything-v2/checkpoints/`. To download it [here](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pt) if needed.

Generate maps for all scenes:

```bash
docker compose run --rm dngaussian python scripts/generate_dav2_depth.py
```

Output: `data/llff/<scene>/depth-maps-DAV2/`.

## Train and evaluate

Generate the corresponding depth maps first. Train fern with either predictor:

```bash
# Default MiDaS DPT maps:
docker compose run --rm dngaussian \
  bash scripts/run_llff.sh data/llff/fern output/llff/fern/default 0

# DAV2 maps:
docker compose run --rm dngaussian \
  bash scripts/run_llff.sh data/llff/fern output/llff/fern/dav2 0 --dav2
```

You can use this to run default MiDaS DPT for all scenes:

```bash
for scene in fern flower fortress horns leaves orchids room trex; do
  docker compose run --rm dngaussian bash scripts/run_llff.sh \
    "data/llff/$scene" "output/llff/$scene/<default|dav2>" 0 || break 2
done
```

