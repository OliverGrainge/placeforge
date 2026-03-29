# HPC / SLURM Setup Guide

## 1. Install dependencies

```bash
# If you need a specific CUDA build of torch, install it first, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## 2. Pre-download model weights

Compute nodes have no internet access, so pretrained weights must be cached
on a login node first. This populates `~/.cache/torch/hub` by default:

```bash
python prefetch_models.py
```

This downloads DINOv2 (vitb14), ResNet-50 (ImageNet), and SALAD.

If your home directory is not shared with compute nodes, set `TORCH_HOME` to
a shared filesystem path before running (and export the same in your slurm
script):

```bash
export TORCH_HOME=/shared/path/torch_cache
python prefetch_models.py
```

## 3. Configure dataset paths

All dataset paths are controlled by three environment variables, loaded from
`.env` by the CLI or overridable as exports.

Edit `.env` (or export in your slurm script) to match your cluster layout:

```
PLACEFORGE_RAW_DIR=/path/to/raw/datasets
PLACEFORGE_FEATURE_STORE_DIR=/path/to/feature_store
PLACEFORGE_PROCESSED_DIR=/path/to/processed
```

The data pipelines expect raw SF-XL images at:

    $PLACEFORGE_RAW_DIR/sf_xl/processed/train/

If your cluster stores them at a different subpath, update `SF_XL_PATH` in
`src/datapipelines/train.py` (line 22) to match.

## 4. Update the SLURM script

Edit `submit_train_job.sh`:

- Uncomment and update the environment activation line (conda/venv)
- Uncomment and set the `PLACEFORGE_*` path exports if not using `.env`
- If `TORCH_HOME` was customised in step 2, uncomment that export too
- `WANDB_MODE=offline` is already set (no internet on compute nodes)

## 5. Submit jobs

Single config:
```bash
sbatch submit_train_job.sh src/configs/sf_xl_classification/cosplace_boq.yaml
```

All configs in a directory:
```bash
bash src/slurm_training_runs.sh src/configs/sf_xl_classification/
```

## 6. Sync wandb logs (after training)

From a login node with internet:
```bash
wandb sync <path/to/wandb/run/directory>
```
