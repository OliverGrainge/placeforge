from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cli
import yaml


class CliDatapipelineTests(unittest.TestCase):
    def test_datapipeline_handler_runs_pipeline_with_empty_context_by_default(
        self,
    ) -> None:
        args = argparse.Namespace(
            name="sf_xl_small",
            context=None,
            list=False,
        )

        with patch("cli.get_pipeline") as get_pipeline:
            pipeline = get_pipeline.return_value

            exit_code = cli._handle_datapipeline(args)

        self.assertEqual(exit_code, 0)
        pipeline.run.assert_called_once_with({}, use_cache=True)

    def test_datapipeline_handler_lists_registered_pipelines(self) -> None:
        args = argparse.Namespace(
            name=None,
            context=None,
            list=True,
        )

        stdout = io.StringIO()
        with (
            patch(
                "cli.list_pipelines",
                side_effect=lambda category: {
                    "train": ("alpha",),
                    "val": (),
                    "test": ("beta",),
                }[category],
            ),
            patch("sys.stdout", stdout),
        ):
            exit_code = cli._handle_datapipeline(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            "train:\n  alpha\nval:\n  <none>\ntest:\n  beta\n",
        )

    def test_datapipeline_requires_name_when_not_listing(self) -> None:
        args = argparse.Namespace(
            name=None,
            context=None,
            list=False,
        )

        with self.assertRaises(SystemExit) as exc:
            cli._handle_datapipeline(args)

        self.assertEqual(
            str(exc.exception), "datapipeline name is required unless --list is used"
        )


class CliTestOutputTests(unittest.TestCase):
    def test_print_test_results_deduplicates_dataloader_metrics(self) -> None:
        results = [
            {
                "test/msls_val/R@1": 0.7408248782157898,
                "test/msls_val/R@5": 0.8333938717842102,
            },
            {
                "test/msls_val/R@1": 0.7408248782157898,
                "test/msls_val/R@5": 0.8333938717842102,
            },
        ]

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            cli._print_test_results(results)

        output = stdout.getvalue()
        self.assertEqual(output.count("msls_val"), 1)
        self.assertEqual(output.count("R@1"), 1)
        self.assertEqual(output.count("R@5"), 1)
        self.assertIn("74.1", output)
        self.assertIn("83.3", output)
        self.assertNotIn("DataLoader", output)

    def test_run_test_suppresses_lightning_table(self) -> None:
        trainer = Mock()
        module = object()
        datamodule = object()
        trainer.test.return_value = [{"test/msls_val/R@1": 0.7408248782157898}]

        with patch("cli._print_test_results") as print_results:
            metrics = cli._run_test_and_print_results(trainer, module, datamodule)

        trainer.test.assert_called_once_with(
            module, datamodule=datamodule, verbose=False
        )
        print_results.assert_called_once_with(trainer.test.return_value)
        self.assertEqual(metrics, {"test/msls_val/R@1": 0.7408248782157898})

    def test_save_checkpoint_test_metrics_writes_yaml_next_to_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.ckpt"
            ckpt_path.write_text("placeholder")

            output_path = cli._save_checkpoint_test_metrics(
                ckpt_path,
                {
                    "test/msls_val/R@1": 0.74,
                    "test/msls_val/R@5": 0.83,
                    "test/pitts30k_val/R@1": 0.64,
                },
            )

            self.assertEqual(output_path, ckpt_path.with_suffix(".test_metrics.yaml"))
            saved = yaml.safe_load(output_path.read_text())
            self.assertEqual(saved["checkpoint"], str(ckpt_path))
            self.assertEqual(saved["metrics"]["msls_val"]["R@1"], 0.74)
            self.assertEqual(saved["metrics"]["msls_val"]["R@5"], 0.83)
            self.assertEqual(saved["metrics"]["pitts30k_val"]["R@1"], 0.64)
            self.assertEqual(saved["average"]["R@1"], 0.69)
            self.assertEqual(saved["average"]["R@5"], 0.83)

    def test_save_checkpoint_test_metrics_skips_non_recall_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.ckpt"
            ckpt_path.write_text("placeholder")

            output_path = cli._save_checkpoint_test_metrics(
                ckpt_path,
                {"test/loss": 1.23},
            )

            self.assertIsNone(output_path)
            self.assertFalse(ckpt_path.with_suffix(".test_metrics.yaml").exists())


class CliTestConfigTests(unittest.TestCase):
    def test_build_parser_accepts_data_config_for_test(self) -> None:
        parser = cli.build_parser()

        args = parser.parse_args(
            [
                "test",
                "src/configs/test/eval.yaml",
                "--data-config",
                "src/configs/test/test_datasets.yaml",
            ]
        )

        self.assertEqual(args.data_config, Path("src/configs/test/test_datasets.yaml"))

    def test_parse_test_module_kwargs_accepts_top_level_architecture(self) -> None:
        module_kwargs = cli._parse_test_module_kwargs(
            {
                "architecture": "eigenplaces",
                "val_recall_ks": [1, 5, 10],
            }
        )

        self.assertEqual(
            module_kwargs,
            {
                "model_name": "eigenplaces",
                "val_recall_ks": [1, 5, 10],
            },
        )

    def test_parse_test_module_kwargs_allows_val_recall_ks_from_data_config(self) -> None:
        module_kwargs = cli._parse_test_module_kwargs(
            {
                "architecture": "eigenplaces",
            },
            {
                "val_recall_ks": [1, 10, 20],
            },
        )

        self.assertEqual(
            module_kwargs,
            {
                "model_name": "eigenplaces",
                "val_recall_ks": [1, 10, 20],
            },
        )

    def test_parse_test_datamodule_kwargs_uses_external_data_config_and_image_size(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            model_config_path = config_dir / "eval.yaml"
            data_config_path = config_dir / "datasets.yaml"
            model_config_path.write_text("placeholder")
            data_config_path.write_text(
                yaml.safe_dump(
                    {
                        "test_dataset_names": ["pitts30k_test", "msls_val"],
                        "batch_size": 16,
                        "num_workers": 4,
                        "val_recall_ks": [1, 10, 20],
                    },
                    sort_keys=False,
                )
            )

            data_config = cli._load_test_data_config(
                model_config={
                    "architecture": "eigenplaces",
                    "image_size": 224,
                    "data_config": "datasets.yaml",
                },
                data_config_path=None,
                model_config_path=model_config_path,
            )
            datamodule_kwargs = cli._parse_test_datamodule_kwargs(
                model_config={
                    "architecture": "eigenplaces",
                    "image_size": 224,
                    "data_config": "datasets.yaml",
                },
                data_config=data_config,
            )

        self.assertEqual(datamodule_kwargs["test_dataset_names"], ["pitts30k_test", "msls_val"])
        self.assertEqual(datamodule_kwargs["batch_size"], 16)
        self.assertEqual(datamodule_kwargs["num_workers"], 4)
        self.assertEqual(datamodule_kwargs["val_dataset_names"], [])
        self.assertEqual(type(datamodule_kwargs["test_transform"]).__name__, "EvalTransform")
        self.assertNotIn("val_recall_ks", datamodule_kwargs)

    def test_parse_test_datamodule_kwargs_falls_back_to_embedded_datamodule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_config_path = Path(tmpdir) / "eval.yaml"
            model_config_path.write_text("placeholder")

            data_config = cli._load_test_data_config(
                model_config={
                    "image_size": 224,
                    "datamodule": {
                        "test_dataset_names": ["tokyo247_test"],
                        "batch_size": 8,
                        "num_workers": 2,
                    },
                },
                data_config_path=None,
                model_config_path=model_config_path,
            )
            datamodule_kwargs = cli._parse_test_datamodule_kwargs(
                model_config={
                    "image_size": 224,
                    "datamodule": {
                        "test_dataset_names": ["tokyo247_test"],
                        "batch_size": 8,
                        "num_workers": 2,
                    },
                },
                data_config=data_config,
            )

        self.assertEqual(datamodule_kwargs["test_dataset_names"], ["tokyo247_test"])
        self.assertEqual(datamodule_kwargs["batch_size"], 8)
        self.assertEqual(datamodule_kwargs["num_workers"], 2)
        self.assertEqual(type(datamodule_kwargs["test_transform"]).__name__, "EvalTransform")


if __name__ == "__main__":
    unittest.main()
