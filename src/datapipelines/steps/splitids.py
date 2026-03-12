from __future__ import annotations

from typing import Any

from .base import BaseStep


class AssignSplitIdsStep(BaseStep):
    """Assign sequential uint32 IDs to query and database image DataFrames.

    Adds a ``qid`` column to the queries frame and a ``dbid`` column to the
    database frame.  Both are zero-indexed and stored as ``uint32``, matching
    the VPR validation dataset schema.
    """

    def __init__(
        self,
        *,
        queries_context_key: str = "queries",
        database_context_key: str = "database",
        query_id_column: str = "qid",
        database_id_column: str = "dbid",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.queries_context_key = queries_context_key
        self.database_context_key = database_context_key
        self.query_id_column = query_id_column
        self.database_id_column = database_id_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "AssignSplitIdsStep requires pandas. Install it with `pip install pandas`."
            ) from exc

        for key in (self.queries_context_key, self.database_context_key):
            if key not in context:
                raise KeyError(f"Context is missing dataframe at key {key!r}")

        queries = context[self.queries_context_key].copy()
        database = context[self.database_context_key].copy()

        with self.progress(total=2, desc="assign split ids") as progress:
            queries[self.query_id_column] = pd.array(range(len(queries)), dtype="uint32")
            progress.update(1)
            database[self.database_id_column] = pd.array(range(len(database)), dtype="uint32")
            progress.update(1)

        context = dict(context)
        context[self.queries_context_key] = queries
        context[self.database_context_key] = database
        return context
