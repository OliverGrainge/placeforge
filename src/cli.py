from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from datapipelines import get_pipeline, list_pipelines




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placeforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", type=Path, help="Path to a YAML config file")
    train_parser.set_defaults(handler=_handle_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.set_defaults(handler=_handle_eval)

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
        choices=["train", "val"],
        help="Filter pipelines by category (train or val)",
    )
    datapipeline_parser.set_defaults(handler=_handle_datapipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
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
        datamodule_kwargs["train_transform"] = get_transform(transform_name, **transform_kwargs)

    val_transform_kwargs: dict[str, Any] | None = datamodule_kwargs.pop("val_transform", None)
    if val_transform_kwargs is not None:
        val_transform_name = val_transform_kwargs.pop("name")
        datamodule_kwargs["val_transform"] = get_transform(val_transform_name, **val_transform_kwargs)

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
    num_supergroups = len(set(train_ds.valid_supergroup_ids))

    print()
    print("=" * 52)
    print(f"  Train: {datamodule.train_dataset_name}")
    print(f"    places:      {num_places:,}")
    print(f"    images/place:{images_per_place:>6}")
    print(f"    supergroups: {num_supergroups:,}")
    print(f"    batch size:  {datamodule.batch_size:,}  "
          f"({datamodule.places_per_batch} places × {images_per_place} images)")

    for name, ds in zip(val_names, val_datasets):
        print(f"  Val: {name}")
        print(f"    queries:  {ds.num_queries:,}")
        print(f"    database: {ds.num_database:,}")

    print("=" * 52)
    print()


def _handle_eval(args: argparse.Namespace) -> int:
    raise NotImplementedError("`eval` is not implemented yet")


def _handle_datapipeline(args: argparse.Namespace) -> int:
    if args.list:
        category: str | None = getattr(args, "category", None)
        categories = [category] if category else ["train", "val"]
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


if __name__ == "__main__":
    raise SystemExit(main())
