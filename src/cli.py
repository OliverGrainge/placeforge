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
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.ckpt if it exists",
    )
    train_parser.set_defaults(handler=_handle_train)

    test_parser = subparsers.add_parser("test", help="Test a model from a checkpoint")
    test_parser.add_argument(
        "path",
        type=Path,
        help="Path to a config YAML, checkpoint directory, or .ckpt file",
    )
    test_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file (inferred from checkpoint dir if not given)",
    )
    test_parser.add_argument(
        "--best",
        action="store_true",
        help="Use best.ckpt instead of last.ckpt when loading from a directory",
    )
    test_parser.set_defaults(handler=_handle_test)

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

    dataloader_parser = subparsers.add_parser(
        "dataloader",
        help="Visualise sample batches and measure dataloader throughput",
    )
    dataloader_parser.add_argument("config", type=Path, help="Path to a YAML config file")
    dataloader_parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of batches to iterate for throughput measurement (default: 50)",
    )
    dataloader_parser.add_argument(
        "--save-batches",
        type=int,
        default=3,
        help="Number of batches to save as figures (default: 3)",
    )
    dataloader_parser.set_defaults(handler=_handle_dataloader)

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


def _build_loggers(logger_config: dict | list[dict]) -> list[Any]:
    """Instantiate one or more PyTorch Lightning loggers from config.

    Accepts a single logger dict or a list of logger dicts.
    Each dict must have a ``type`` key (wandb, csv, or tensorboard).
    """
    from pytorch_lightning import loggers as pl_loggers

    loggers_map = {
        "wandb": pl_loggers.WandbLogger,
        "csv": pl_loggers.CSVLogger,
        "tensorboard": pl_loggers.TensorBoardLogger,
    }

    entries = logger_config if isinstance(logger_config, list) else [logger_config]
    result = []
    for entry in entries:
        entry = dict(entry)  # shallow copy so we don't mutate config
        name = entry.pop("type")
        entry.setdefault("save_dir", str(_LOGS_DIR))
        cls = loggers_map.get(name)
        if cls is None:
            raise ValueError(f"Unknown logger '{name}'. Choose from: {list(loggers_map.keys())}")
        result.append(cls(**entry))
    return result


_LOGS_DIR = Path(__file__).parent.parent / "logs"
_CONFIGS_DIR = Path(__file__).parent / "configs"
_CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"


def _checkpoint_dir(config_path: Path) -> Path:
    """Return checkpoint dir mirroring the config's position under configs/."""
    try:
        rel = config_path.resolve().relative_to(_CONFIGS_DIR.resolve())
    except ValueError:
        rel = Path(config_path.stem)
    return _CHECKPOINTS_DIR / rel.with_suffix("")


def _handle_train(args: argparse.Namespace) -> int:
    import torch
    import pytorch_lightning as pl

    torch.set_float32_matmul_precision("high")

    from datamodules import get_datamodule
    from modules import get_module

    config = yaml.safe_load(args.config.read_text())

    trainer_kwargs: dict[str, Any] = config.get("trainer", {})
    module_kwargs: dict[str, Any] = config.get("module", {})

    module_name = module_kwargs.pop("name")
    datamodule_name, datamodule_kwargs = _parse_datamodule_kwargs(config)

    trainer_kwargs.setdefault("default_root_dir", str(_LOGS_DIR))

    logger_config: dict | list[dict] | None = config.get("logger")
    # Backwards compat: top-level "wandb" key still works
    if logger_config is None and config.get("wandb") is not None:
        logger_config = {**config["wandb"], "type": "wandb"}

    if logger_config is not None:
        trainer_kwargs["logger"] = _build_loggers(logger_config)

    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_dir = _checkpoint_dir(args.config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[Any] = trainer_kwargs.pop("callbacks", [])
    callbacks += [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="val/R@1",
            mode="max",
            save_top_k=1,
            save_last=False,
            enable_version_counter=False,
        ),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="last",
            save_top_k=0,
            save_last=True,
            enable_version_counter=False,
        ),
    ]
    trainer_kwargs["callbacks"] = callbacks

    module = get_module(module_name, **module_kwargs)
    if config.get("compile", False):
        compile_kwargs = config["compile"] if isinstance(config["compile"], dict) else {}
        module = torch.compile(module, **compile_kwargs)
    datamodule = get_datamodule(datamodule_name, **datamodule_kwargs)

    datamodule.setup("fit")
    _print_dataset_summary(datamodule)

    ckpt_path: str | None = None
    if args.resume:
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            print(f"Resuming from {last_ckpt}")
            ckpt_path = str(last_ckpt)
        else:
            print(f"No checkpoint found at {last_ckpt}, starting from scratch")

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    if datamodule.test_dataset_names:
        datamodule.setup("test")
        trainer.test(module, datamodule=datamodule)

    return 0


def _config_from_checkpoint_dir(ckpt_dir: Path) -> Path | None:
    """Inverse of _checkpoint_dir: derive config path from a checkpoint directory."""
    try:
        rel = ckpt_dir.resolve().relative_to(_CHECKPOINTS_DIR.resolve())
    except ValueError:
        return None
    return _CONFIGS_DIR / rel.with_suffix(".yaml")


def _handle_test(args: argparse.Namespace) -> int:
    import torch
    import pytorch_lightning as pl

    torch.set_float32_matmul_precision("high")

    from datamodules import get_datamodule
    from modules import get_module

    # Resolve checkpoint file and config
    path_arg = args.path.resolve()
    ckpt_name = "best.ckpt" if args.best else "last.ckpt"

    if path_arg.suffix in (".yaml", ".yml"):
        # User passed a config file — derive checkpoint dir from it
        config_path = path_arg
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return 1
        ckpt_dir = _checkpoint_dir(config_path)
        ckpt_file = ckpt_dir / ckpt_name
    elif path_arg.is_dir():
        ckpt_dir = path_arg
        ckpt_file = ckpt_dir / ckpt_name
        config_path = args.config or _config_from_checkpoint_dir(ckpt_dir)
    else:
        ckpt_file = path_arg
        ckpt_dir = path_arg.parent
        config_path = args.config or _config_from_checkpoint_dir(ckpt_dir)

    if not ckpt_file.exists():
        print(f"Checkpoint not found: {ckpt_file}")
        return 1

    if config_path is None or not config_path.exists():
        print(
            f"Cannot infer config from {ckpt_dir}. "
            "Pass --config path/to/config.yaml explicitly."
        )
        return 1

    config = yaml.safe_load(config_path.read_text())

    trainer_kwargs: dict[str, Any] = config.get("trainer", {})
    module_kwargs: dict[str, Any] = config.get("module", {})

    module_name = module_kwargs.pop("name")
    datamodule_name, datamodule_kwargs = _parse_datamodule_kwargs(config)

    trainer_kwargs.setdefault("default_root_dir", str(_LOGS_DIR))
    trainer_kwargs.pop("callbacks", None)

    module = get_module(module_name, **module_kwargs)
    datamodule = get_datamodule(datamodule_name, **datamodule_kwargs)

    if not datamodule.test_dataset_names:
        print("No test datasets configured in this config.")
        return 1

    datamodule.setup("test")

    print(f"Testing with checkpoint: {ckpt_file}")

    import torch
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.test(module, datamodule=datamodule, ckpt_path=str(ckpt_file))

    return 0


def _print_dataset_summary(datamodule: Any) -> None:
    train_ds = datamodule._train_dataset
    val_datasets = datamodule._val_datasets
    val_names = datamodule.val_dataset_names

    num_places = train_ds.num_places
    num_images = train_ds.num_images

    print()
    print("=" * 52)
    print(f"  Train: {datamodule.train_dataset_name}")
    print(f"    places:      {num_places:,}")
    print(f"    images:      {num_images:,}")
    if hasattr(train_ds, "num_supergroups"):
        num_supergroups = train_ds.num_supergroups
        sg_place_counts = list(train_ds.supergroup_num_places.values())
        sg_min = min(sg_place_counts)
        sg_max = max(sg_place_counts)
        sg_mean = sum(sg_place_counts) / len(sg_place_counts)
        print(f"    supergroups: {num_supergroups:,}")
        print(f"    places/sg:   {sg_mean:.1f}  (min {sg_min}  max {sg_max})")
    if datamodule.dataset_type == "classification":
        print(f"    batch size:  {datamodule.batch_size:,}  (image-level)")
    else:
        images_per_place = train_ds.images_per_place
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


def _parse_datamodule_kwargs(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract datamodule name and kwargs from a loaded YAML config.

    Resolves nested transform configs into instantiated transform objects.
    Returns (datamodule_name, datamodule_kwargs).
    """
    from modules import get_transform

    datamodule_kwargs: dict[str, Any] = config.get("datamodule", {})
    datamodule_name = datamodule_kwargs.pop("name")

    for config_key, kwarg_key in [
        ("transform", "train_transform"),
        ("val_transform", "val_transform"),
        ("test_transform", "test_transform"),
    ]:
        transform_kwargs: dict[str, Any] | None = datamodule_kwargs.pop(config_key, None)
        if transform_kwargs is not None:
            transform_name = transform_kwargs.pop("name")
            datamodule_kwargs[kwarg_key] = get_transform(transform_name, **transform_kwargs)

    return datamodule_name, datamodule_kwargs


def _handle_dataloader(args: argparse.Namespace) -> int:
    import os
    import time

    import torch
    import numpy as np

    torch.set_float32_matmul_precision("high")

    from datamodules import get_datamodule

    config = yaml.safe_load(args.config.read_text())
    datamodule_name, datamodule_kwargs = _parse_datamodule_kwargs(config)

    datamodule = get_datamodule(datamodule_name, **datamodule_kwargs)
    datamodule.setup("fit")

    _print_dataset_summary(datamodule)

    loader = datamodule.train_dataloader()

    # ── Save sample batch figures ────────────────────────────────────────────
    output_dir = (
        Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
        / "train"
        / datamodule.train_dataset_name
        / "dataloader_samples"
    )
    if args.save_batches > 0:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)

        # ImageNet denorm constants
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        loader_iter = iter(loader)
        for batch_idx in range(args.save_batches):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            images = batch["images"]  # (B, C, H, W)
            place_ids = batch.get("place_ids", batch.get("labels"))  # (B,)
            sg_id = batch["supergroup_id"].item() if "supergroup_id" in batch else None

            n_images = images.shape[0]
            n_cols = min(n_images, 8)
            n_rows = (n_images + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 2.5, n_rows * 2.5),
                squeeze=False,
            )

            for i in range(n_images):
                row, col = divmod(i, n_cols)
                ax = axes[row][col]
                img = images[i] * std + mean
                img = img.clamp(0, 1).permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(f"pid={place_ids[i].item()}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])

            for i in range(n_images, n_rows * n_cols):
                row, col = divmod(i, n_cols)
                axes[row][col].set_visible(False)

            title = f"batch {batch_idx}  |  {n_images} images"
            if sg_id is not None:
                title = f"batch {batch_idx}  |  supergroup {sg_id}  |  {n_images} images"
            fig.suptitle(title, fontsize=10)
            fig.tight_layout()
            fig.savefig(output_dir / f"batch_{batch_idx}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {min(args.save_batches, batch_idx + 1)} batch figures to {output_dir}")

    # ── Measure throughput ───────────────────────────────────────────────────
    from tqdm import tqdm

    loader_iter = iter(loader)
    total_images = 0
    pbar = tqdm(range(args.num_batches), desc="Throughput", unit="batch")
    t_start = time.perf_counter()
    for i in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        total_images += batch["images"].shape[0]
        elapsed = time.perf_counter() - t_start
        pbar.set_postfix_str(
            f"{total_images / elapsed:.0f} img/s, "
            f"{elapsed / (i + 1) * 1000:.1f} ms/batch"
        )
    elapsed = time.perf_counter() - t_start
    batches_done = i + 1
    pbar.close()

    print(f"\n  {batches_done} batches, {total_images:,} images in {elapsed:.2f}s")
    print(f"  {total_images / elapsed:.1f} images/s")
    print(f"  {batches_done / elapsed:.1f} batches/s")
    print(f"  {elapsed / batches_done * 1000:.1f} ms/batch")

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
    pipeline.run(context, use_cache=True)
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
