"""anyembed_tui - a tiny terminal UI for anyembed.

Launch with `anyembed` (no arguments) or `anyembed tui`. Type text or a
file/folder path into the box, then press Enter to search the local DB or
Ctrl+A to embed and add it. Embedding runs in a background thread so the
UI stays responsive while the 7B model loads on first use.
"""

from __future__ import annotations

import os
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Input, Static


class AnyEmbedTUI(App):
    TITLE = "anyembed"
    SUB_TITLE = "Enter = search · Ctrl+A = add · Ctrl+Q = quit"

    CSS = """
    Input { margin: 1 1 0 1; }
    #status { height: 1; padding: 0 2; color: $text-muted; }
    DataTable { margin: 0 1; }
    """

    BINDINGS = [
        # priority=True so the shortcut works while the Input box has focus
        Binding("ctrl+a", "add", "Add to DB", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, db=None):
        super().__init__()
        self._db_instance = db

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Type text, or a file/folder path…", id="query")
        yield Static("Model loads on first search/add (7B — be patient).", id="status")
        yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("similarity", "modality", "source / text")
        self.query_one(Input).focus()

    # -- helpers ----------------------------------------------------------

    def _db(self):
        if self._db_instance is None:
            import anyembed

            self._db_instance = anyembed.AnyEmbedDB()
        return self._db_instance

    def _query_text(self) -> Optional[str]:
        value = self.query_one(Input).value.strip()
        return value or None

    def set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def show_hits(self, hits: list[dict]) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for hit in hits:
            doc = hit["document"].replace("\n", " ")
            table.add_row(
                f"{hit['similarity']:.3f}",
                hit["metadata"].get("modality", "?"),
                doc[:100] + ("…" if len(doc) > 100 else ""),
            )
        self.set_status(f"{len(hits)} result(s)" if hits else "No results — add something first (Ctrl+A).")

    # -- events / actions --------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if query:
            self.set_status(f"Searching for {query!r}…")
            self.run_search(query)

    def action_add(self) -> None:
        item = self._query_text()
        if item:
            self.set_status(f"Embedding {item!r}…")
            self.run_add(item)

    # -- background work ---------------------------------------------------

    @work(thread=True, exclusive=True, group="embed")
    def run_search(self, query: str) -> None:
        try:
            hits = self._db().search(query, top_k=10)
        except Exception as exc:
            self.call_from_thread(self.set_status, f"Search failed: {exc}")
            return
        self.call_from_thread(self.show_hits, hits)

    @work(thread=True, exclusive=True, group="embed")
    def run_add(self, item: str) -> None:
        try:
            if os.path.isdir(os.path.expanduser(item)):
                results = self._db().add_folder(os.path.expanduser(item))
                message = f"Added {len(results)} file(s) from {item}"
            else:
                id = self._db().add(item)
                message = f"Added {item!r} -> {id}"
        except Exception as exc:
            self.call_from_thread(self.set_status, f"Add failed: {exc}")
            return
        self.call_from_thread(self.set_status, message)


def main() -> None:
    AnyEmbedTUI().run()


if __name__ == "__main__":
    main()
