from __future__ import annotations

import argparse
import io
import unittest
from unittest.mock import Mock, patch

import cli


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
            cli._run_test_and_print_results(trainer, module, datamodule)

        trainer.test.assert_called_once_with(
            module, datamodule=datamodule, verbose=False
        )
        print_results.assert_called_once_with(trainer.test.return_value)


if __name__ == "__main__":
    unittest.main()
