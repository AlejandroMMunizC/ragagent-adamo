"""RAGAGENT - desktop entry point.

Builds the FAISS index over the documents in `DOCS/`, wires up the LangChain
agent, and exposes it to a small pywebview frontend (`index.html`).

Run:

    python app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import webview

from ragagent import config
from ragagent.agent import build_agent
from ragagent.rag import RagIndex


HERE = Path(__file__).resolve().parent


def _build_runtime():
    """Build the index + agent. Surfaces friendly errors if something is missing."""
    index = RagIndex().build()
    stats = index.stats()
    print(f"[RAGAGENT] Indexed {stats['documents']} documents -> "
          f"{stats['chunks']} chunks.")

    if index.is_empty:
        print(
            "[RAGAGENT] Warning: no documents found in DOCS/. "
            "The agent will only be able to answer using web search.",
            file=sys.stderr,
        )

    agent = build_agent(index)
    return index, agent


class API:
    """Methods exposed to the JavaScript frontend via pywebview."""

    def __init__(self, agent):
        self._agent = agent

    # The frontend keeps the original Spanish method names so the existing
    # index.html keeps working without changes.
    def preguntar_al_agente(self, question: str, k: int = config.DEFAULT_K) -> str:
        try:
            print(f"[RAGAGENT] Running agent (k={k}) for: {question!r}")
            return self._agent.run(question)
        except Exception as exc:  # noqa: BLE001
            print(f"[RAGAGENT] Agent error: {exc}", file=sys.stderr)
            return f"Internal error: {exc}"

    def guardar_respuesta_completa(self, payload: dict) -> bool:
        config.LOGS_DIR.mkdir(exist_ok=True)
        timestamp = payload["timestamp"].replace(":", "-").replace(".", "-")
        target = config.LOGS_DIR / f"log_{timestamp}.json"
        target.write_text(
            json.dumps(payload, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        return True

    def cerrar_aplicacion(self) -> None:
        # Force exit so background threads (FAISS, sentence-transformers) don't hang.
        os._exit(0)


def main() -> None:
    _, agent = _build_runtime()
    api = API(agent)

    html_path = HERE / "index.html"
    webview.create_window(
        "RAGAGENT - Agentic RAG assistant",
        url=html_path.as_uri(),
        width=1000,
        height=800,
        js_api=api,
    )
    webview.start()


if __name__ == "__main__":
    main()
