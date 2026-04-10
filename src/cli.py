from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv
from prettytable import PrettyTable

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
        help="Path to a model/eval YAML config file (inferred from checkpoint dir if not given)",
    )
    test_parser.add_argument(
        "--data-config",
        type=Path,
        default=None,
        help="Path to a YAML config describing test datasets, batch size, workers, and transforms",
    )
    test_parser.add_argument(
        "--best",
        action="store_true",
        help="Use best.ckpt instead of last.ckpt when loading from a directory",
    )
    test_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit path to a .ckpt file (overrides checkpoint resolution from path)",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _build_logger(logger_config: dict) -> Any:
    """Instantiate a TensorBoard logger from config."""
    from pytorch_lightning.loggers import TensorBoardLogger

    config = dict(logger_config)
    config.pop("type", None)
    config.setdefault("save_dir", str(_LOGS_DIR))
    return TensorBoardLogger(**config)


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

    from datamodules import CuraVPRTrainDataModule
    from modules import CuraVPRLightningModule

    config = yaml.safe_load(args.config.read_text())

    trainer_kwargs: dict[str, Any] = config.get("trainer", {})
    module_kwargs: dict[str, Any] = config.get("module", {})
    datamodule_kwargs = _parse_datamodule_kwargs(config)

    trainer_kwargs.setdefault("default_root_dir", str(_LOGS_DIR))

    logger_config: dict | None = config.get("logger")
    if logger_config is not None:
        trainer_kwargs["logger"] = _build_logger(logger_config)

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

    module = CuraVPRLightningModule(**module_kwargs)
    if config.get("compile", False):
        compile_kwargs = config["compile"] if isinstance(config["compile"], dict) else {}
        module = torch.compile(module, **compile_kwargs)
    datamodule = CuraVPRTrainDataModule(**datamodule_kwargs)

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


def _load_checkpoint_into_model(model: Any, ckpt_path: Path) -> None:
    """Robustly load a checkpoint into *model* (not the Lightning module).

    Strategy
    --------
    1. Load the file; if it contains a ``state_dict`` key, use that,
       otherwise treat the whole dict as the state dict.
    2. Detect and strip/add a systematic key prefix so that the checkpoint
       keys align with ``model.state_dict()`` keys.  Common mismatches:
       ``model.`` prefix from a Lightning module, ``module.`` from DDP, etc.
    3. Fail **hard** if any of the model's keys are still missing after
       alignment — we never silently ignore missing weights.
    """
    import torch

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # --- 1. extract state dict -------------------------------------------
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict: dict[str, Any] = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(
            f"Checkpoint at {ckpt_path} is not a dict and has no 'state_dict' key."
        )

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # --- 2. try to align keys --------------------------------------------
    # If keys already match, skip prefix detection.
    if not model_keys <= ckpt_keys:
        state_dict = _align_state_dict(state_dict, model_keys)
        ckpt_keys = set(state_dict.keys())

    # --- 3. strict check: every model key must be present ----------------
    missing = model_keys - ckpt_keys
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} model key(s) after alignment. "
            f"First 10: {sorted(missing)[:10]}"
        )

    extra = ckpt_keys - model_keys
    if extra:
        print(f"  [checkpoint] Ignoring {len(extra)} extra key(s) not in model.")

    model.load_state_dict({k: state_dict[k] for k in model_keys}, strict=True)
    print(f"  [checkpoint] Loaded {len(model_keys)} parameters into model.")


def _align_state_dict(
    state_dict: dict[str, Any], model_keys: set[str],
) -> dict[str, Any]:
    """Try to fix a systematic prefix mismatch between *state_dict* and *model_keys*.

    Handles cases like:
    - ckpt has ``model.dino.blocks.0...`` but model expects ``dino.blocks.0...``
    - ckpt has ``dino.blocks.0...`` but model expects ``model.dino.blocks.0...``
    - ckpt has ``module.model.dino...`` (DDP wrapping) etc.
    """
    ckpt_keys = list(state_dict.keys())

    # Pick a reference model key to detect the prefix relationship.
    ref_model_key = min(model_keys)  # deterministic pick

    # Case A: checkpoint keys have an extra prefix to strip.
    for ckpt_key in ckpt_keys:
        if ckpt_key.endswith(ref_model_key):
            prefix = ckpt_key[: len(ckpt_key) - len(ref_model_key)]
            # Verify this prefix explains most keys.
            stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if model_keys <= set(stripped.keys()):
                print(f"  [checkpoint] Stripped prefix '{prefix}' from checkpoint keys.")
                return stripped

    # Case B: model keys have an extra prefix — checkpoint is "inner".
    ref_ckpt_key = min(ckpt_keys)
    for model_key in sorted(model_keys):
        if model_key.endswith(ref_ckpt_key):
            prefix = model_key[: len(model_key) - len(ref_ckpt_key)]
            added = {prefix + k: v for k, v in state_dict.items()}
            if model_keys <= set(added.keys()):
                print(f"  [checkpoint] Added prefix '{prefix}' to checkpoint keys.")
                return added

    # No systematic fix found — return as-is and let the caller error out.
    return state_dict


def _handle_test(args: argparse.Namespace) -> int:
    import torch
    import pytorch_lightning as pl

    torch.set_float32_matmul_precision("high")

    from datamodules import CuraVPRTrainDataModule
    from modules import CuraVPRLightningModule

    # Resolve checkpoint file and config
    path_arg = args.path.resolve()
    ckpt_name = "best.ckpt" if args.best else "last.ckpt"
    use_robust_load = False

    ckpt_file = None
    config_path = None
    config: dict[str, Any] = {}

    if args.checkpoint is not None:
        # Explicit --checkpoint arg takes top priority
        ckpt_file = args.checkpoint.resolve()
        if path_arg.suffix in (".yaml", ".yml"):
            config_path = path_arg
        else:
            config_path = args.config or _config_from_checkpoint_dir(ckpt_file.parent)
        use_robust_load = True
    elif path_arg.suffix in (".yaml", ".yml"):
        # User passed a config file — check for checkpoint key inside it
        config_path = path_arg
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return 1
        config = _load_yaml_config(config_path)
        checkpoint_value = config.get("checkpoint")
        if checkpoint_value is not None:
            ckpt_file = _resolve_config_path(Path(checkpoint_value).expanduser(), config_path)
            use_robust_load = True
        else:
            # No checkpoint in config — check if there's one in the default dir
            ckpt_dir = _checkpoint_dir(config_path)
            candidate = ckpt_dir / ckpt_name
            if candidate.exists():
                ckpt_file = candidate
    elif path_arg.is_dir():
        ckpt_dir = path_arg
        ckpt_file = ckpt_dir / ckpt_name
        config_path = args.config or _config_from_checkpoint_dir(ckpt_dir)
    else:
        ckpt_file = path_arg
        ckpt_dir = path_arg.parent
        config_path = args.config or _config_from_checkpoint_dir(ckpt_dir)

    if ckpt_file is not None and not ckpt_file.exists():
        print(f"Checkpoint not found: {ckpt_file}")
        return 1

    if config_path is None or not config_path.exists():
        if ckpt_file is not None:
            print(
                f"Cannot infer config from {ckpt_file.parent}. "
                "Pass --config path/to/config.yaml explicitly."
            )
        else:
            print("No config file found. Pass a YAML config as the path argument.")
        return 1

    if not config:
        config = _load_yaml_config(config_path)

    trainer_kwargs: dict[str, Any] = config.get("trainer", {})
    test_data_config = _load_test_data_config(
        model_config=config,
        data_config_path=args.data_config,
        model_config_path=config_path,
    )
    module_kwargs = _parse_test_module_kwargs(config, test_data_config)
    datamodule_kwargs = _parse_test_datamodule_kwargs(
        model_config=config,
        data_config=test_data_config,
    )

    trainer_kwargs.setdefault("default_root_dir", str(_LOGS_DIR))
    trainer_kwargs.pop("callbacks", None)

    module = CuraVPRLightningModule(**module_kwargs)
    datamodule = CuraVPRTrainDataModule(**datamodule_kwargs)

    if not datamodule.test_dataset_names:
        print("No test datasets configured in this config.")
        return 1

    datamodule.setup("test")

    import torch

    if ckpt_file is None:
        # No checkpoint — run with the model's own pretrained weights (e.g. baselines)
        print("Testing without checkpoint (using model's built-in weights)")
        trainer = pl.Trainer(**trainer_kwargs)
        _run_test_and_print_results(trainer, module, datamodule)
    elif use_robust_load:
        print(f"Testing with checkpoint: {ckpt_file}")
        # Robust manual loading: load directly into the model, not the lightning module
        _load_checkpoint_into_model(module.model, ckpt_file)
        trainer = pl.Trainer(**trainer_kwargs)
        metrics = _run_test_and_print_results(trainer, module, datamodule)
        _save_checkpoint_test_metrics(ckpt_file, metrics)
    else:
        print(f"Testing with checkpoint: {ckpt_file}")
        _orig_torch_load = torch.load
        torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})
        trainer = pl.Trainer(**trainer_kwargs)
        metrics = _run_test_and_print_results(
            trainer, module, datamodule, ckpt_path=str(ckpt_file)
        )
        _save_checkpoint_test_metrics(ckpt_file, metrics)

    return 0


def _run_test_and_print_results(
    trainer: Any,
    module: Any,
    datamodule: Any,
    ckpt_path: str | None = None,
) -> dict[str, float]:
    test_kwargs: dict[str, Any] = {"datamodule": datamodule, "verbose": False}
    if ckpt_path is not None:
        test_kwargs["ckpt_path"] = ckpt_path

    results = trainer.test(module, **test_kwargs)
    metrics = _unique_test_metrics(results)
    _print_test_results(results)
    return metrics


def _save_checkpoint_test_metrics(
    ckpt_path: Path,
    metrics: Mapping[str, float],
) -> Path | None:
    recall_metrics = {
        name: float(value) for name, value in metrics.items() if _is_recall_metric(name)
    }
    if not recall_metrics:
        return None

    output_path = ckpt_path.with_suffix(".test_metrics.yaml")
    grouped_metrics: dict[str, dict[str, float]] = {}
    average_metrics: dict[str, float] = {}

    datasets, recall_names = _group_recall_metrics_by_dataset(recall_metrics)
    for dataset, recalls in datasets.items():
        grouped_metrics[dataset] = {recall: float(value) for recall, value in recalls.items()}

    for recall_name in recall_names:
        values = [
            recalls[recall_name]
            for recalls in datasets.values()
            if recall_name in recalls
        ]
        if values:
            average_metrics[recall_name] = sum(values) / len(values)

    output = {
        "checkpoint": str(ckpt_path),
        "metrics": grouped_metrics,
    }
    if average_metrics:
        output["average"] = average_metrics

    output_path.write_text(yaml.safe_dump(output, sort_keys=False))
    print(f"Saved test metrics to {output_path}")
    return output_path


def _print_test_results(results: list[Mapping[str, float]]) -> None:
    metrics = _unique_test_metrics(results)
    if not metrics:
        return

    all_recalls = all(_is_recall_metric(name) for name in metrics)
    table = PrettyTable()
    table.align = "l"

    if all_recalls:
        table.title = "Test metrics (%)"
        datasets, recall_metrics = _group_recall_test_metrics(metrics)
        table.field_names = ["Dataset", *recall_metrics]
        for recall_metric in recall_metrics:
            table.align[recall_metric] = "r"
        for dataset, recalls in datasets.items():
            table.add_row(
                [dataset, *(recalls.get(metric, "-") for metric in recall_metrics)]
            )
        # Average row
        avg_row = ["Average"]
        for metric in recall_metrics:
            values = [
                float(recalls[metric])
                for recalls in datasets.values()
                if metric in recalls and recalls[metric] != "-"
            ]
            avg_row.append(f"{sum(values) / len(values):.1f}" if values else "-")
        table.add_row(avg_row)
    else:
        table.field_names = ["Metric", "Value"]
        table.align["Value"] = "r"
        for name, value in metrics.items():
            table.add_row([name, _format_test_metric_value(name, value)])

    print()
    print(table)
    print()


def _unique_test_metrics(results: list[Mapping[str, float]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for result in results:
        for name, value in result.items():
            metrics.setdefault(name, float(value))
    return metrics


def _format_test_metric_value(name: str, value: float) -> str:
    if _is_recall_metric(name):
        return f"{value * 100:.1f}"
    return f"{value:.1f}"


def _is_recall_metric(name: str) -> bool:
    return "/R@" in name


def _group_recall_test_metrics(
    metrics: Mapping[str, float],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    datasets, recall_metrics = _group_recall_metrics_by_dataset(metrics)
    formatted = {
        dataset: {
            recall: _format_test_metric_value(f"test/{dataset}/{recall}", value)
            for recall, value in recalls.items()
        }
        for dataset, recalls in datasets.items()
    }
    return formatted, recall_metrics


def _group_recall_metrics_by_dataset(
    metrics: Mapping[str, float],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    datasets: dict[str, dict[str, str]] = {}
    recall_metrics: list[str] = []

    for name, value in metrics.items():
        dataset, recall = _split_test_recall_metric_name(name)
        datasets.setdefault(dataset, {})[recall] = float(value)
        if recall not in recall_metrics:
            recall_metrics.append(recall)

    return datasets, recall_metrics


def _split_test_recall_metric_name(name: str) -> tuple[str, str]:
    dataset, recall = name.rsplit("/", maxsplit=1)
    return dataset.removeprefix("test/"), recall


def _print_dataset_summary(datamodule: Any) -> None:
    import numpy as np

    train_ds = datamodule._train_dataset
    val_datasets = datamodule._val_datasets
    val_names = datamodule.val_dataset_names

    # Use _full_df when available (epoch-cycling datamodules like cosplace/eigenplaces
    # only load one supergroup into _train_dataset at a time).
    full_df = getattr(datamodule, "_full_df", None)

    if full_df is not None:
        num_images = len(full_df)
    elif hasattr(train_ds, "num_images"):
        num_images = train_ds.num_images
    else:
        num_images = len(train_ds)

    print()
    print("=" * 52)
    print(f"  Train: {datamodule.train_dataset_name}")
    print(f"    images:       {num_images:,}")

    if full_df is not None and "place_id" in full_df.columns:
        num_places = full_df["place_id"].nunique()
        print(f"    places:       {num_places:,}")
        images_per_place = full_df.groupby("place_id").size()
        print(
            f"    images/place: avg={images_per_place.mean():.1f}  "
            f"min={images_per_place.min()}  max={images_per_place.max()}  "
            f"std={images_per_place.std():.1f}"
        )
    elif hasattr(train_ds, "num_places"):
        print(f"    places:       {train_ds.num_places:,}")
        if hasattr(train_ds, "df"):
            images_per_place = train_ds.df.groupby("place_id").size()
            print(
                f"    images/place: avg={images_per_place.mean():.1f}  "
                f"min={images_per_place.min()}  max={images_per_place.max()}  "
                f"std={images_per_place.std():.1f}"
            )

    if full_df is not None and "supergroup_id" in full_df.columns:
        sg_ids = sorted(full_df["supergroup_id"].unique())
        print(f"    supergroups:  {len(sg_ids)}")
        if "place_id" in full_df.columns:
            sg_place_counts = full_df.groupby("supergroup_id")["place_id"].nunique().values
            print(
                f"    places/sg:    avg={sg_place_counts.mean():.1f}  "
                f"min={sg_place_counts.min()}  max={sg_place_counts.max()}  "
                f"std={sg_place_counts.std():.1f}"
            )
        images_per_sg = full_df.groupby("supergroup_id").size().values
        print(
            f"    images/sg:    avg={images_per_sg.mean():.1f}  "
            f"min={images_per_sg.min()}  max={images_per_sg.max()}  "
            f"std={images_per_sg.std():.1f}"
        )
    elif hasattr(train_ds, "num_supergroups"):
        print(f"    supergroups:  {train_ds.num_supergroups}")
        sg_place_counts = np.array(list(train_ds.supergroup_num_places.values()))
        print(
            f"    places/sg:    avg={sg_place_counts.mean():.1f}  "
            f"min={sg_place_counts.min()}  max={sg_place_counts.max()}  "
            f"std={sg_place_counts.std():.1f}"
        )
        if hasattr(train_ds, "df") and "supergroup_id" in train_ds.df.columns:
            images_per_sg = train_ds.df.groupby("supergroup_id").size().values
            print(
                f"    images/sg:    avg={images_per_sg.mean():.1f}  "
                f"min={images_per_sg.min()}  max={images_per_sg.max()}  "
                f"std={images_per_sg.std():.1f}"
            )

    if hasattr(train_ds, "images_per_place"):
        images_per_place_k = train_ds.images_per_place
        print(
            f"    batch size:   {datamodule.batch_size * images_per_place_k:,}  "
            f"({datamodule.batch_size} places × {images_per_place_k} images)"
        )
    else:
        print(f"    batch size:   {datamodule.batch_size:,}")

    for name, ds in zip(val_names, val_datasets):
        print(f"  Val: {name}")
        print(f"    queries:  {ds.num_queries:,}")
        print(f"    database: {ds.num_database:,}")

    print("=" * 52)
    print()


def _parse_datamodule_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract datamodule kwargs from a loaded YAML config.

    Resolves nested transform configs into instantiated transform objects.
    """
    from modules import TrainTransform, EvalTransform

    _TRANSFORMS = {"train": TrainTransform, "eval": EvalTransform}

    if "datamodule" in config:
        datamodule_kwargs: dict[str, Any] = dict(config["datamodule"])
    else:
        datamodule_kwargs = dict(config)

    for config_key, kwarg_key in [
        ("transform", "train_transform"),
        ("val_transform", "val_transform"),
        ("test_transform", "test_transform"),
    ]:
        transform_kwargs: dict[str, Any] | None = datamodule_kwargs.pop(config_key, None)
        if transform_kwargs is not None:
            transform_name = transform_kwargs.pop("name")
            transform_cls = _TRANSFORMS[transform_name]
            datamodule_kwargs[kwarg_key] = transform_cls(**transform_kwargs)

    return datamodule_kwargs


def _load_yaml_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def _resolve_config_path(path: Path, config_path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _parse_test_module_kwargs(
    config: Mapping[str, Any],
    data_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    module_kwargs = dict(config.get("module", {}))

    if "architecture" in config and "model_name" not in module_kwargs:
        module_kwargs["model_name"] = config["architecture"]
    if "val_recall_ks" in config and "val_recall_ks" not in module_kwargs:
        module_kwargs["val_recall_ks"] = config["val_recall_ks"]
    if (
        data_config is not None
        and "val_recall_ks" in data_config
        and "val_recall_ks" not in config.get("module", {})
    ):
        module_kwargs["val_recall_ks"] = data_config["val_recall_ks"]

    return module_kwargs


def _load_test_data_config(
    *,
    model_config: Mapping[str, Any],
    data_config_path: Path | None,
    model_config_path: Path,
) -> dict[str, Any]:
    resolved_data_config_path = _resolve_test_data_config_path(
        model_config=model_config,
        data_config_path=data_config_path,
        model_config_path=model_config_path,
    )

    if resolved_data_config_path is not None:
        return _load_yaml_config(resolved_data_config_path)

    return {"datamodule": dict(model_config.get("datamodule", {}))}


def _parse_test_datamodule_kwargs(
    *,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    data_config = dict(data_config)

    if "datamodule" in data_config:
        datamodule_config = dict(data_config["datamodule"])
        data_config = {**data_config, "datamodule": datamodule_config}
    else:
        datamodule_config = {
            key: value
            for key, value in data_config.items()
            if key not in {"val_recall_ks", "architecture", "checkpoint", "data_config", "test_data_config", "trainer", "module", "image_size"}
        }
        data_config = datamodule_config

    if "image_size" in model_config and "test_transform" not in datamodule_config:
        datamodule_config["test_transform"] = {
            "name": "eval",
            "image_size": model_config["image_size"],
        }

    datamodule_kwargs = _parse_datamodule_kwargs(data_config)

    datamodule_kwargs.setdefault("val_dataset_names", [])

    return datamodule_kwargs


def _resolve_test_data_config_path(
    *,
    model_config: Mapping[str, Any],
    data_config_path: Path | None,
    model_config_path: Path,
) -> Path | None:
    if data_config_path is not None:
        return data_config_path.resolve()

    configured_path = model_config.get("data_config") or model_config.get("test_data_config")
    if configured_path is None:
        return None

    return _resolve_config_path(Path(configured_path).expanduser(), model_config_path)


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


if __name__ == "__main__":
    raise SystemExit(main())
