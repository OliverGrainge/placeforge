# PlaceForge

A PyTorch Lightning framework for training and evaluating visual place recognition (VPR) models. PlaceForge supports modern vision backbones (DINOv2, ResNet), multiple aggregation methods (GeM, BoQ, SALAD), and contrastive metric learning on standard geo-localization benchmarks.

## Features

- **Modular architecture** -- mix-and-match backbones and aggregation heads
- **Scalable data pipeline** -- process raw datasets into training-ready parquet files with geographic place clustering
- **Contrastive training** -- MultiSimilarityLoss with hard-negative mining via supergroup-aware batch sampling
- **Standard benchmarks** -- evaluate on Pitts30k, MSLS, Tokyo247, Nordland, SVOX, and SF-XL
- **Pretrained baselines** -- one-command evaluation of MegaLoc, EigenPlaces, MixVPR, SALAD, and more
- **HPC ready** -- SLURM submission scripts and offline model prefetching

## Setup

Requires Python >= 3.10.

```bash
pip install -r requirements.txt
python prefetch_models.py  # optional: pre-download model weights for offline nodes
```

Configure dataset paths in a `.env` file:

```
PLACEFORGE_RAW_DIR=/path/to/raw/datasets
PLACEFORGE_PROCESSED_DIR=/path/to/processed
```

## Usage

### Process a dataset

```bash
placeforge datapipeline sf_xl_train
placeforge datapipeline pitts30k_val
```

This reads raw images, computes embeddings, assigns geographic place IDs via spatial clustering, creates supergroups (k-means), and exports parquet files.

### Train a model

```bash
placeforge train src/configs/train/sf_xl_boq.yaml
placeforge train src/configs/train/sf_xl_boq.yaml --resume  # resume from last checkpoint
```

Training configs are YAML files under `src/configs/train/` specifying the model, learning rate, dataset, batch size, and more. See the existing configs for examples.

### Evaluate

```bash
placeforge test src/configs/train/sf_xl_boq.yaml --best
```

Reports Recall@1/5/10 on the configured test datasets.

### HPC / SLURM

```bash
sbatch submit_train_job.sh src/configs/train/sf_xl_boq.yaml
```

See `HPC_SETUP.md` for cluster-specific setup.

## Supported Models

| Backbone | Aggregation | Config example |
|---|---|---|
| DINOv2-ViT-B/14 | GeM | `sf_xl_gem.yaml` |
| DINOv2-ViT-B/14 | BoQ | `sf_xl_boq.yaml` |
| DINOv2-ViT-B/14 | SALAD | `sf_xl_salad.yaml` |
| ResNet-18 | GeM | (lightweight / sanity checks) |

## Supported Datasets

**Training:** SF-XL, Pittsburgh 30k, MSLS, GSVCities (and combinations)

**Evaluation:** Pitts30k, MSLS, Tokyo247, Nordland, SVOX, SF-XL-small

## Project Structure

```
src/
  cli.py                  # Entry point (train, test, datapipeline)
  modules/                # Lightning modules and model definitions
    models/               # DINOv2, ResNet, SelaVPR++, baselines
    transforms/           # Train/eval image augmentations
  datamodules/            # Lightning data modules and datasets
  datapipelines/          # Dataset processing pipelines and steps
  configs/                # YAML training and evaluation configs
```

## License

See repository for license details.
