from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

from datapipelines import get_pipeline, list_pipelines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placeforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", type=Path, help="Path to a YAML config file")
    train_parser.set_defaults(handler=_handle_train)

    analyse_parser = subparsers.add_parser(
        "analyse",
        help="Analyse a training dataset: sample figure + intra-class variation",
    )
    analyse_parser.add_argument("dataset_name", help="Name of the training dataset")
    analyse_parser.add_argument(
        "--num-places",
        type=int,
        default=16,
        help="Number of places to sample (default: 16)",
    )
    analyse_parser.add_argument(
        "--images-per-place",
        type=int,
        default=4,
        help="Images per place to show (default: 4)",
    )
    analyse_parser.add_argument(
        "--image-embedding-name",
        default=None,
        help="Feature-store name for image embeddings (defaults to dataset_name)",
    )
    analyse_parser.add_argument(
        "--place-embedding-name",
        default=None,
        help="Feature-store name for place embeddings (defaults to dataset_name)",
    )
    analyse_parser.set_defaults(handler=_handle_analyse)

    datapipeline_parser = subparsers.add_parser(
        "datapipeline",
        help="Run a registered data pipeline",
    )
    datapipeline_parser.add_argument(
        "name",
        nargs="?",
        help="Registered pipeline name",
    )
    datapipeline_parser.add_argument(
        "--context",
        type=Path,
        help="Optional path to a Python file defining INITIAL_CONTEXT as a dict",
    )
    datapipeline_parser.add_argument(
        "--list",
        action="store_true",
        help="List registered pipelines and exit",
    )
    datapipeline_parser.add_argument(
        "--category",
        choices=["train", "val", "test"],
        help="Filter pipelines by category (train, val, or test)",
    )
    datapipeline_parser.set_defaults(handler=_handle_datapipeline)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare dataset statistics in a terminal table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compare training datasets side-by-side.\n"
            "Reads summary.json and stats.json from PLACEFORGE_PROCESSED_DIR/train/<name>/.\n"
            "\n"
            "Columns:\n"
            "  Places        Number of unique place clusters\n"
            "  Images        Total images in the dataset\n"
            "  Img/place     Mean number of images per place\n"
            "  Img/pl range  Min–max images per place\n"
            "  Supergroups   Number of supergroups (higher-level spatial clusters)\n"
            "  Pl/SG         Mean number of places per supergroup\n"
            "  Pl/SG range   Min–max places per supergroup\n"
            "\n"
            "  Intra *       Cosine distance between images of the SAME place\n"
            "                (lower = same-place images are more similar / tighter cluster)\n"
            "  Intra mean    Mean of per-place mean pairwise cosine distances\n"
            "  Intra p5/p95  5th / 95th percentile of per-place mean cosine distances\n"
            "  Intra std     Std-dev of per-place mean cosine distances\n"
            "\n"
            "  Inter *       Cosine distance between places of the SAME supergroup\n"
            "                (higher = places are more spread out / better separated)\n"
            "  Inter mean    Mean of per-supergroup mean pairwise cosine distances\n"
            "  Inter p5/p95  5th / 95th percentile of per-supergroup mean cosine distances\n"
            "  Inter std     Std-dev of per-supergroup mean cosine distances\n"
            "\n"
            "  All cosine distances are in [0, 2]; 0 = identical, 1 = orthogonal.\n"
            "  Intra/inter columns are omitted if no dataset in the comparison has that data.\n"
        ),
    )
    compare_parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset names to compare",
    )
    compare_parser.set_defaults(handler=_handle_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


_LOGS_DIR = Path(__file__).parent.parent / "logs"


def _handle_train(args: argparse.Namespace) -> int:
    import torch
    import pytorch_lightning as pl

    torch.set_float32_matmul_precision("high")

    from datamodules import get_datamodule
    from modules import get_module, get_transform

    config = yaml.safe_load(args.config.read_text())

    trainer_kwargs: dict[str, Any] = config.get("trainer", {})
    module_kwargs: dict[str, Any] = config.get("module", {})
    datamodule_kwargs: dict[str, Any] = config.get("datamodule", {})

    module_name = module_kwargs.pop("name")
    datamodule_name = datamodule_kwargs.pop("name")

    trainer_kwargs.setdefault("default_root_dir", str(_LOGS_DIR))

    wandb_kwargs: dict[str, Any] | None = config.get("wandb")
    if wandb_kwargs is not None:
        from pytorch_lightning.loggers import WandbLogger

        wandb_kwargs.setdefault("save_dir", str(_LOGS_DIR))
        trainer_kwargs["logger"] = WandbLogger(**wandb_kwargs)

    transform_kwargs: dict[str, Any] | None = datamodule_kwargs.pop("transform", None)
    if transform_kwargs is not None:
        transform_name = transform_kwargs.pop("name")
        datamodule_kwargs["train_transform"] = get_transform(
            transform_name, **transform_kwargs
        )

    val_transform_kwargs: dict[str, Any] | None = datamodule_kwargs.pop(
        "val_transform", None
    )
    if val_transform_kwargs is not None:
        val_transform_name = val_transform_kwargs.pop("name")
        datamodule_kwargs["val_transform"] = get_transform(
            val_transform_name, **val_transform_kwargs
        )

    module = get_module(module_name, **module_kwargs)
    datamodule = get_datamodule(datamodule_name, **datamodule_kwargs)

    datamodule.setup("fit")
    _print_dataset_summary(datamodule)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule)
    return 0


def _print_dataset_summary(datamodule: Any) -> None:
    train_ds = datamodule._train_dataset
    val_datasets = datamodule._val_datasets
    val_names = datamodule.val_dataset_names

    num_places = len(train_ds)
    images_per_place = train_ds.images_per_place
    num_supergroups = train_ds.num_supergroups

    print()
    print("=" * 52)
    print(f"  Train: {datamodule.train_dataset_name}")
    print(f"    places:      {num_places:,}")
    print(f"    images/place:{images_per_place:>6}")
    print(f"    supergroups: {num_supergroups:,}")
    print(
        f"    batch size:  {datamodule.batch_size * images_per_place:,}  "
        f"({datamodule.batch_size} places × {images_per_place} images)"
    )

    for name, ds in zip(val_names, val_datasets):
        print(f"  Val: {name}")
        print(f"    queries:  {ds.num_queries:,}")
        print(f"    database: {ds.num_database:,}")

    print("=" * 52)
    print()


def _handle_analyse(args: argparse.Namespace) -> int:
    import json
    import os
    import random
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datamodules.datasets import TrainDataset
    from datapipelines.steps.util import EmbeddingCache

    ds = TrainDataset(args.dataset_name, images_per_place=args.images_per_place)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    sg_counts: dict = {}
    for place_id in ds.place_ids:
        sg = ds.place_id_to_supergroup[place_id]
        sg_counts[sg] = sg_counts.get(sg, 0) + 1

    img_counts = [len(paths) for paths in ds.place_id_to_paths.values()]
    places_per_sg = list(sg_counts.values())

    print()
    print("=" * 52)
    print(f"  Dataset: {args.dataset_name}")
    print(f"    places:           {ds.num_places:,}")
    print(f"    images:           {ds.num_images:,}")
    print(f"    supergroups:      {ds.num_supergroups:,}")
    print(
        f"    images/place:     avg={sum(img_counts)/len(img_counts):.1f}  "
        f"min={min(img_counts)}  max={max(img_counts)}"
    )
    print(
        f"    places/supergroup avg={sum(places_per_sg)/len(places_per_sg):.1f}  "
        f"min={min(places_per_sg)}  max={max(places_per_sg)}"
    )
    print("=" * 52)
    print()

    # ── Sample batch figures → analysis/sample_batches/ ──────────────────────
    out_dir = ds.dataset_dir
    batches_dir = out_dir / "sample_batches"
    batches_dir.mkdir(exist_ok=True)

    num_places = min(args.num_places, ds.num_places)
    n_cols = args.images_per_place
    n_batches = 10

    sg_to_indices = ds._supergroup_to_indices
    eligible_sgs = [sg for sg, idxs in sg_to_indices.items() if len(idxs) >= 2]

    for batch_i in range(n_batches):
        # Mirror the real batching strategy: all places from the same supergroup
        sg_id = random.choice(eligible_sgs)
        indices = sg_to_indices[sg_id]
        sampled_indices = random.sample(indices, min(num_places, len(indices)))
        n_rows = len(sampled_indices)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2.2, n_rows * 2.2),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.12)

        for row, idx in enumerate(sampled_indices):
            sample = ds[idx]
            place_id = sample["place_id"]
            images = sample["images"]
            for col in range(n_cols):
                ax = axes[row][col]
                img = images[col]
                img = img.permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(
                        f"p{place_id}",
                        fontsize=6,
                        rotation=0,
                        labelpad=28,
                        va="center",
                    )

        for col in range(n_cols):
            axes[0][col].set_title(f"img {col + 1}", fontsize=8)
        fig.suptitle(
            f"{args.dataset_name} — batch {batch_i + 1}/{n_batches} — supergroup {sg_id} — {n_rows} places × {n_cols} images",
            fontsize=10,
            y=1.002,
        )

        batch_path = batches_dir / f"batch_{batch_i + 1:02d}.png"
        fig.savefig(batch_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {n_batches} sample batches → {batches_dir}/")

    # ── Shared helpers ────────────────────────────────────────────────────────
    image_embedding_name = args.image_embedding_name or args.dataset_name
    place_embedding_name = args.place_embedding_name or args.dataset_name
    feature_store_dir = Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])

    def _summary(series: pd.Series) -> dict:
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "p5": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    def _hist(ax, data, title, xlabel, ylabel, color):
        ax.hist(
            data, bins=min(60, len(data)), color=color, edgecolor="none", alpha=0.85
        )
        ax.axvline(
            data.mean(),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"mean={data.mean():.3f}",
        )
        ax.axvline(
            data.median(),
            color="grey",
            linestyle=":",
            linewidth=1.0,
            label=f"median={data.median():.3f}",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    image_cache = EmbeddingCache(
        feature_store_dir / "embedding" / "image" / image_embedding_name
    )

    if not image_cache.exists:
        print(
            f"\n  Skipping variation analysis: embeddings not found at {image_cache.cache_dir}"
        )
        print(
            f"  Run the datapipeline with ComputeEmbeddingStep, or pass --image-embedding-name.\n"
        )
        return 0

    # ── Intra-class variation ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    df = ds.df.reset_index()
    emb_index = image_cache.load_index().set_index("id")["row"]
    emb_mmap = image_cache.mmap()

    df = df.join(emb_index.rename("emb_row"), on="image_id", how="inner")
    missing = len(ds.df) - df["emb_row"].notna().sum()
    if missing:
        print(f"  Warning: {missing} images have no embedding — skipping them.")
    df = df.dropna(subset=["emb_row"])
    df["emb_row"] = df["emb_row"].astype(int)

    # Bulk-load all needed embeddings in one mmap read, then move to device
    df = df.reset_index(drop=True)
    all_emb_rows = df["emb_row"].values
    all_vecs = torch.from_numpy(emb_mmap[all_emb_rows].astype(np.float32))  # keep on CPU
    df["local_row"] = np.arange(len(df))

    # Collect place groups, sorted by size for efficient padding
    place_groups: list[tuple] = []
    skipped_single = 0
    for place_id, group in df.groupby("place_id"):
        n = len(group)
        if n < 2:
            skipped_single += 1
            continue
        place_groups.append((
            int(place_id),
            int(group["supergroup_id"].iloc[0]),
            group["local_row"].values,
        ))
    place_groups.sort(key=lambda x: len(x[2]))

    per_place: list[dict] = []
    INTRA_BATCH = 64

    for batch_start in tqdm(
        range(0, len(place_groups), INTRA_BATCH),
        desc="Intra-class distances",
        unit="batch",
        total=(len(place_groups) + INTRA_BATCH - 1) // INTRA_BATCH,
    ):
        batch = place_groups[batch_start : batch_start + INTRA_BATCH]
        B = len(batch)
        max_n = len(batch[-1][2])  # sorted by size, last is largest
        D = all_vecs.shape[1]

        padded = torch.zeros(B, max_n, D, device=device)
        ns = []
        for j, (_, _, rows) in enumerate(batch):
            n = len(rows)
            ns.append(n)
            v = all_vecs[rows].to(device)
            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)
            padded[j, :n] = v

        dist_mat = (1.0 - torch.bmm(padded, padded.transpose(1, 2))).cpu().numpy()

        for j, (place_id, sg_id, _) in enumerate(batch):
            n = ns[j]
            triu_i, triu_j = np.triu_indices(n, k=1)
            pairwise_dist = dist_mat[j, triu_i, triu_j]
            per_place.append(
                {
                    "place_id": place_id,
                    "supergroup_id": sg_id,
                    "n_images": n,
                    "mean_cos_dist": float(pairwise_dist.mean()),
                    "max_cos_dist": float(pairwise_dist.max()),
                    "min_cos_dist": float(pairwise_dist.min()),
                    "std_cos_dist": float(pairwise_dist.std()),
                }
            )

    if skipped_single:
        print(f"  Skipped {skipped_single} single-image places.")

    place_df = pd.DataFrame(per_place)
    intra_summary = {
        "dataset": args.dataset_name,
        "image_embedding_name": image_embedding_name,
        "n_places_analysed": len(place_df),
        "n_places_skipped_single_image": skipped_single,
        "mean_cos_dist": _summary(place_df["mean_cos_dist"]),
        "max_cos_dist": _summary(place_df["max_cos_dist"]),
        "min_cos_dist": _summary(place_df["min_cos_dist"]),
        "std_cos_dist": _summary(place_df["std_cos_dist"]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Intra-class variation — {args.dataset_name} ({len(place_df):,} places)",
        fontsize=12,
    )
    _hist(
        axes[0, 0],
        place_df["mean_cos_dist"],
        "Mean cosine distance",
        "mean pairwise cos-dist",
        "# places",
        "#4C72B0",
    )
    _hist(
        axes[0, 1],
        place_df["max_cos_dist"],
        "Max cosine distance",
        "max pairwise cos-dist",
        "# places",
        "#C44E52",
    )
    _hist(
        axes[0, 2],
        place_df["min_cos_dist"],
        "Min cosine distance",
        "min pairwise cos-dist",
        "# places",
        "#55A868",
    )
    _hist(
        axes[1, 0],
        place_df["std_cos_dist"],
        "Std cosine distance",
        "std pairwise cos-dist",
        "# places",
        "#8172B2",
    )
    sorted_vals = np.sort(place_df["mean_cos_dist"].values)
    axes[1, 1].plot(
        sorted_vals,
        np.arange(1, len(sorted_vals) + 1) / len(sorted_vals),
        color="#4C72B0",
        linewidth=1.5,
    )
    axes[1, 1].set_title("CDF — mean cosine distance", fontsize=10)
    axes[1, 1].set_xlabel("mean pairwise cos-dist", fontsize=8)
    axes[1, 1].set_ylabel("cumulative fraction", fontsize=8)
    axes[1, 1].tick_params(labelsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].scatter(
        place_df["n_images"],
        place_df["mean_cos_dist"],
        alpha=0.3,
        s=4,
        color="#4C72B0",
        rasterized=True,
    )
    axes[1, 2].set_title("# images vs mean variation", fontsize=10)
    axes[1, 2].set_xlabel("images per place", fontsize=8)
    axes[1, 2].set_ylabel("mean cosine distance", fontsize=8)
    axes[1, 2].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "intra_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved intra distributions → {out_dir / 'intra_distributions.png'}")

    n_sg = place_df["supergroup_id"].nunique()
    if n_sg <= 128:
        sg_groups = (
            place_df.groupby("supergroup_id")["mean_cos_dist"].apply(list).sort_index()
        )
        fig2, ax2 = plt.subplots(figsize=(max(8, n_sg * 0.35), 5))
        ax2.boxplot(
            sg_groups.values,
            tick_labels=[str(s) for s in sg_groups.index],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": "#4C72B0", "alpha": 0.6},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.4},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        ax2.set_title(
            f"Intra-class cosine distance by supergroup — {args.dataset_name}",
            fontsize=10,
        )
        ax2.set_xlabel("supergroup_id", fontsize=8)
        ax2.set_ylabel("mean cosine distance", fontsize=8)
        ax2.tick_params(axis="x", labelsize=6, rotation=90)
        ax2.tick_params(axis="y", labelsize=7)
        ax2.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / "intra_by_supergroup.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved intra supergroup plot → {out_dir / 'intra_by_supergroup.png'}")

    s = intra_summary["mean_cos_dist"]
    print()
    print("=" * 52)
    print(f"  Intra-class variation ({args.dataset_name})")
    print(f"    places:  {len(place_df):,}   embedding: {image_embedding_name}")
    print(f"    mean cos-dist:  mean={s['mean']:.4f}  std={s['std']:.4f}")
    print(f"                    max={s['max']:.4f}  median={s['median']:.4f}")
    print(f"                    p5={s['p5']:.4f}   p95={s['p95']:.4f}")
    print("=" * 52)
    print()

    # ── Inter-class variation ─────────────────────────────────────────────────
    place_cache = EmbeddingCache(
        feature_store_dir / "embedding" / "place" / place_embedding_name
    )

    if not place_cache.npy_path.exists():
        print(
            f"  Skipping inter-class analysis: place embeddings not found at {place_cache.cache_dir}\n"
        )
        # Still write the combined stats with only intra
        with open(out_dir / "stats.json", "w") as f:
            json.dump(
                {"intra": {"summary": intra_summary, "per_place": per_place}},
                f,
                indent=2,
            )
        print(f"  Saved stats → {out_dir / 'stats.json'}")
        return 0

    place_emb_mmap = place_cache.mmap()
    place_index = place_cache.load_index().set_index("id")["row"]

    sg_to_places: dict = {}
    for pid in ds.place_ids:
        if pid not in place_index:
            continue
        sg_to_places.setdefault(ds.place_id_to_supergroup[pid], []).append(pid)

    per_sg: list[dict] = []
    sg_pairwise_dists: dict[int, list] = {}
    skipped_single_sg = 0

    for sg_id, place_ids in tqdm(
        sg_to_places.items(), desc="Inter-class distances", unit="supergroup"
    ):
        n = len(place_ids)
        if n < 2:
            skipped_single_sg += 1
            continue
        rows = place_index.loc[place_ids].values
        vecs = torch.from_numpy(place_emb_mmap[rows].astype(np.float32)).to(device)
        vecs = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        triu_i, triu_j = torch.triu_indices(n, n, offset=1, device=device)
        pairwise_dist = (1.0 - vecs @ vecs.T)[triu_i, triu_j].cpu().numpy()
        sg_pairwise_dists[int(sg_id)] = pairwise_dist.tolist()
        per_sg.append(
            {
                "supergroup_id": int(sg_id),
                "n_places": int(n),
                "mean_cos_dist": float(pairwise_dist.mean()),
                "max_cos_dist": float(pairwise_dist.max()),
                "min_cos_dist": float(pairwise_dist.min()),
                "std_cos_dist": float(pairwise_dist.std()),
            }
        )

    if skipped_single_sg:
        print(f"  Skipped {skipped_single_sg} single-place supergroups.")

    sg_df = pd.DataFrame(per_sg)
    inter_summary = {
        "dataset": args.dataset_name,
        "place_embedding_name": place_embedding_name,
        "n_supergroups_analysed": len(sg_df),
        "n_supergroups_skipped_single_place": skipped_single_sg,
        "mean_cos_dist": _summary(sg_df["mean_cos_dist"]),
        "max_cos_dist": _summary(sg_df["max_cos_dist"]),
        "min_cos_dist": _summary(sg_df["min_cos_dist"]),
        "std_cos_dist": _summary(sg_df["std_cos_dist"]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Inter-class variation — {args.dataset_name} ({len(sg_df):,} supergroups)",
        fontsize=12,
    )
    _hist(
        axes[0, 0],
        sg_df["mean_cos_dist"],
        "Mean inter-class cosine distance",
        "mean pairwise cos-dist",
        "# supergroups",
        "#4C72B0",
    )
    _hist(
        axes[0, 1],
        sg_df["max_cos_dist"],
        "Max cosine distance (most separable)",
        "max pairwise cos-dist",
        "# supergroups",
        "#C44E52",
    )
    _hist(
        axes[0, 2],
        sg_df["min_cos_dist"],
        "Min cosine distance (most confusable)",
        "min pairwise cos-dist",
        "# supergroups",
        "#55A868",
    )
    _hist(
        axes[1, 0],
        sg_df["std_cos_dist"],
        "Std cosine distance",
        "std pairwise cos-dist",
        "# supergroups",
        "#8172B2",
    )
    sorted_vals = np.sort(sg_df["mean_cos_dist"].values)
    axes[1, 1].plot(
        sorted_vals,
        np.arange(1, len(sorted_vals) + 1) / len(sorted_vals),
        color="#4C72B0",
        linewidth=1.5,
    )
    axes[1, 1].set_title("CDF — mean inter-class cosine distance", fontsize=10)
    axes[1, 1].set_xlabel("mean pairwise cos-dist", fontsize=8)
    axes[1, 1].set_ylabel("cumulative fraction", fontsize=8)
    axes[1, 1].tick_params(labelsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].scatter(
        sg_df["n_places"], sg_df["mean_cos_dist"], alpha=0.5, s=12, color="#4C72B0"
    )
    axes[1, 2].set_title("# places vs mean separation", fontsize=10)
    axes[1, 2].set_xlabel("places per supergroup", fontsize=8)
    axes[1, 2].set_ylabel("mean cosine distance", fontsize=8)
    axes[1, 2].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "inter_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved inter distributions → {out_dir / 'inter_distributions.png'}")

    n_sg = len(sg_pairwise_dists)
    if n_sg <= 128:
        sorted_sg = sorted(sg_pairwise_dists.keys())
        fig3, ax3 = plt.subplots(figsize=(max(8, n_sg * 0.35), 5))
        ax3.boxplot(
            [sg_pairwise_dists[s] for s in sorted_sg],
            tick_labels=[str(s) for s in sorted_sg],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": "#C44E52", "alpha": 0.6},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.4},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        ax3.set_title(
            f"Inter-class cosine distance by supergroup — {args.dataset_name}",
            fontsize=10,
        )
        ax3.set_xlabel("supergroup_id", fontsize=8)
        ax3.set_ylabel("pairwise cosine distance", fontsize=8)
        ax3.tick_params(axis="x", labelsize=6, rotation=90)
        ax3.tick_params(axis="y", labelsize=7)
        ax3.grid(axis="y", alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(out_dir / "inter_by_supergroup.png", dpi=120, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved inter supergroup plot → {out_dir / 'inter_by_supergroup.png'}")

    # Combined stats JSON
    with open(out_dir / "stats.json", "w") as f:
        json.dump(
            {
                "intra": {"summary": intra_summary, "per_place": per_place},
                "inter": {"summary": inter_summary, "per_supergroup": per_sg},
            },
            f,
            indent=2,
        )
    print(f"  Saved stats → {out_dir / 'stats.json'}")

    si = inter_summary["mean_cos_dist"]
    print()
    print("=" * 52)
    print(f"  Inter-class variation ({args.dataset_name})")
    print(f"    supergroups: {len(sg_df):,}   embedding: {place_embedding_name}")
    print(f"    mean cos-dist:  mean={si['mean']:.4f}  std={si['std']:.4f}")
    print(f"                    max={si['max']:.4f}  median={si['median']:.4f}")
    print(f"                    p5={si['p5']:.4f}   p95={si['p95']:.4f}")
    print("=" * 52)
    print()

    return 0


def _handle_datapipeline(args: argparse.Namespace) -> int:
    if args.list:
        category: str | None = getattr(args, "category", None)
        categories = [category] if category else ["train", "val", "test"]
        for cat in categories:
            names = list_pipelines(category=cat)
            print(f"{cat}:")
            for pipeline_name in names:
                print(f"  {pipeline_name}")
            if not names:
                print("  <none>")
        return 0

    if args.name is None:
        raise SystemExit("datapipeline name is required unless --list is used")

    pipeline = get_pipeline(args.name)
    context = _load_initial_context(args.context) if args.context else {}
    pipeline.run(context)
    return 0


def _load_initial_context(path: Path) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    source = path.read_text()
    exec(compile(source, str(path), "exec"), namespace)

    initial_context = namespace.get("INITIAL_CONTEXT")
    if initial_context is None:
        raise ValueError(f"{path} does not define INITIAL_CONTEXT")
    if not isinstance(initial_context, dict):
        raise TypeError(f"INITIAL_CONTEXT in {path} must be a dict")

    return initial_context


def _handle_compare(args: argparse.Namespace) -> int:
    import json
    import os

    processed_dir = Path(os.environ["PLACEFORGE_PROCESSED_DIR"])

    # ── Collect rows ──────────────────────────────────────────────────────────
    rows: list[dict] = []
    for name in args.datasets:
        base = processed_dir / "train" / name
        summary_path = base / "summary.json"
        stats_path = base / "stats.json"

        if not summary_path.exists():
            print(f"  Warning: summary.json not found for '{name}' — skipping")
            continue

        summary = json.loads(summary_path.read_text())
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

        n_places = summary["num_places"]
        n_images = summary["num_images"]
        n_supergroups = summary["num_supergroups"]
        img_per_place_mean = summary["images_per_place"]["mean"]
        img_per_place_min = summary["images_per_place"]["min"]
        img_per_place_max = summary["images_per_place"]["max"]

        # places per supergroup from supergroups section
        sg_data = summary.get("supergroups", {})
        if sg_data:
            places_per_sg = [v["num_places"] for v in sg_data.values()]
            avg_places_per_sg = sum(places_per_sg) / len(places_per_sg)
            min_places_per_sg = min(places_per_sg)
            max_places_per_sg = max(places_per_sg)
        else:
            avg_places_per_sg = n_places / n_supergroups if n_supergroups else float("nan")
            min_places_per_sg = float("nan")
            max_places_per_sg = float("nan")

        intra = stats.get("intra", {}).get("summary", {})
        inter = stats.get("inter", {}).get("summary", {})

        row: dict = {
            "name": name,
            "places": n_places,
            "images": n_images,
            "img/place": img_per_place_mean,
            "img/place range": f"{img_per_place_min}–{img_per_place_max}",
            "supergroups": n_supergroups,
            "places/sg": avg_places_per_sg,
            "places/sg range": f"{min_places_per_sg:.0f}–{max_places_per_sg:.0f}" if sg_data else "—",
        }

        if intra:
            m = intra["mean_cos_dist"]
            row["intra mean"] = m["mean"]
            row["intra p5"] = m["p5"]
            row["intra p95"] = m["p95"]
            row["intra std"] = m["std"]
        else:
            row["intra mean"] = row["intra p5"] = row["intra p95"] = row["intra std"] = None

        if inter:
            m = inter["mean_cos_dist"]
            row["inter mean"] = m["mean"]
            row["inter p5"] = m["p5"]
            row["inter p95"] = m["p95"]
            row["inter std"] = m["std"]
        else:
            row["inter mean"] = row["inter p5"] = row["inter p95"] = row["inter std"] = None

        rows.append(row)

    if not rows:
        print("No datasets loaded.")
        return 1

    # ── Format helpers ────────────────────────────────────────────────────────
    def _fmt(val: object, fmt: str = "") -> str:
        if val is None:
            return "—"
        if isinstance(val, float):
            return format(val, fmt) if fmt else f"{val:.4f}"
        return str(val)

    # ── Build column spec: (header, key, fmt_spec) ───────────────────────────
    cols: list[tuple[str, str, str]] = [
        ("Dataset",       "name",           ""),
        ("Places",        "places",         ",d"),
        ("Images",        "images",         ",d"),
        ("Img/place",     "img/place",      ".1f"),
        ("Img/pl range",  "img/place range",""),
        ("Supergroups",   "supergroups",    ",d"),
        ("Pl/SG",         "places/sg",      ".1f"),
        ("Pl/SG range",   "places/sg range",""),
        ("Intra mean",    "intra mean",     ".4f"),
        ("Intra p5",      "intra p5",       ".4f"),
        ("Intra p95",     "intra p95",      ".4f"),
        ("Intra std",     "intra std",      ".4f"),
        ("Inter mean",    "inter mean",     ".4f"),
        ("Inter p5",      "inter p5",       ".4f"),
        ("Inter p95",     "inter p95",      ".4f"),
        ("Inter std",     "inter std",      ".4f"),
    ]

    # Drop columns where all rows are None / "—"
    def _all_missing(key: str) -> bool:
        return all(rows[i].get(key) is None for i in range(len(rows)))

    cols = [(h, k, f) for h, k, f in cols if not _all_missing(k)]

    # Compute column widths
    widths = []
    for header, key, fmt_spec in cols:
        col_vals = [_fmt(r.get(key), fmt_spec) for r in rows]
        widths.append(max(len(header), max(len(v) for v in col_vals)))

    # ── Render ────────────────────────────────────────────────────────────────
    sep = "─"
    def _row_str(values: list[str]) -> str:
        return "  " + "  │  ".join(v.rjust(w) for v, w in zip(values, widths))

    def _sep_line() -> str:
        return "──" + "──┼──".join(sep * w for w in widths)

    print()
    print(_row_str([h for h, _, _ in cols]))
    print(_sep_line())
    for r in rows:
        vals = [_fmt(r.get(k), f) for _, k, f in cols]
        print(_row_str(vals))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
