from __future__ import annotations

from typing import Any

from .base import BaseStep


class RemoveUnmatchedQueriesStep(BaseStep):
    """Remove query images that have no matching database image.

    Reads the queries and matches DataFrames from context and drops any query
    whose ``qid`` does not appear in the matches table.  The matches table is
    left untouched — by definition there are no match rows for unmatched
    queries, so no orphaned rows are created.
    """

    def __init__(
        self,
        *,
        queries_context_key: str = "queries",
        matches_context_key: str = "matches",
        query_id_column: str = "qid",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.queries_context_key = queries_context_key
        self.matches_context_key = matches_context_key
        self.query_id_column = query_id_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        for key in (self.queries_context_key, self.matches_context_key):
            if key not in context:
                raise KeyError(f"Context is missing dataframe at key {key!r}")

        queries = context[self.queries_context_key]
        matches = context[self.matches_context_key]

        matched_qids = set(matches[self.query_id_column].tolist())
        mask = queries[self.query_id_column].isin(matched_qids)
        filtered = queries.loc[mask].reset_index(drop=True)

        n_removed = len(queries) - len(filtered)
        with self.progress(total=1, desc="remove unmatched queries") as progress:
            if n_removed:
                print(f"  removed {n_removed} queries with no database match")
            context = dict(context)
            context[self.queries_context_key] = filtered
            progress.update(1)

        return context
