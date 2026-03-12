from __future__ import annotations

import argparse
import io
import unittest
from unittest.mock import patch

import cli


class CliDatapipelineTests(unittest.TestCase):
    def test_datapipeline_handler_runs_pipeline_with_empty_context_by_default(self) -> None:
        args = argparse.Namespace(
            name="sf_xl_small",
            context=None,
            list=False,
        )

        with patch("cli.get_pipeline") as get_pipeline:
            pipeline = get_pipeline.return_value

            exit_code = cli._handle_datapipeline(args)

        self.assertEqual(exit_code, 0)
        pipeline.run.assert_called_once_with({})

    def test_datapipeline_handler_lists_registered_pipelines(self) -> None:
        args = argparse.Namespace(
            name=None,
            context=None,
            list=True,
        )

        stdout = io.StringIO()
        with patch("cli.list_pipelines", return_value=("alpha", "beta")), patch(
            "sys.stdout", stdout
        ):
            exit_code = cli._handle_datapipeline(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "Available datapipelines:\n- alpha\n- beta\n")

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


if __name__ == "__main__":
    unittest.main()
