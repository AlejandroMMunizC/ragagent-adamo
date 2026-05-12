"""LangChain tools exposed to the agent.

Three tools are registered:

* ``search_documents``   - semantic search over the local FAISS index.
* ``search_web``         - lightweight Bing scraper for fresh information.
* ``send_email``         - send a plain-text email via Gmail SMTP.
"""

from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain.agents import Tool

from . import config
from .rag import RagIndex


# ---------------------------------------------------------------------------
# Tool 1 - local document retrieval
# ---------------------------------------------------------------------------
def make_document_search_tool(index: RagIndex) -> Tool:
    available_docs = ", ".join(sorted(index._chunks_by_doc.keys())) or "(none)"

    def _run(question: str) -> str:
        chunks = index.retrieve(question, k=config.DEFAULT_K)
        if not chunks:
            return "No relevant fragments were found in the local documents."
        return "\n\n".join(chunks)

    return Tool(
        name="search_documents",
        func=_run,
        description=(
            "ALWAYS try this tool FIRST for any factual question. It performs "
            "semantic search over the user's private local document collection "
            f"({available_docs}). These documents describe internal company "
            "information (products, prices, hours, FAQs, etc.) that is NOT on "
            "the public web. Only fall back to `search_web` if this tool returns "
            "'No relevant fragments were found in the local documents.'"
        ),
    )


# ---------------------------------------------------------------------------
# Tool 2 - Bing web search via scraping
# ---------------------------------------------------------------------------
def _scrape_bing(query: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        return f"Search error: HTTP {response.status_code}"

    soup = BeautifulSoup(response.text, "html.parser")
    blocks = soup.select("li.b_algo")
    if not blocks:
        return "No relevant results were found on the web."

    output: List[str] = []
    for block in blocks[:3]:
        link_tag = block.select_one("h2 a")
        desc_tag = block.select_one("p")
        if link_tag:
            title = link_tag.text.strip()
            href = link_tag.get("href", "")
            description = desc_tag.text.strip() if desc_tag else "No description."
            output.append(f"- {title}: {href}\n  {description}")
    return "\n".join(output)


def make_web_search_tool() -> Tool:
    def _run(question: str) -> str:
        try:
            return _scrape_bing(question)
        except Exception as exc:  # noqa: BLE001
            return f"Web search error: {exc}"

    return Tool(
        name="search_web",
        func=_run,
        description=(
            "Search the public web (Bing) for up-to-date information that is "
            "not in the local documents."
        ),
    )


# ---------------------------------------------------------------------------
# Tool 3 - send an email through Gmail SMTP
# ---------------------------------------------------------------------------
_EMAIL_PATTERN = re.compile(
    r"correo a ['\"](?P<to>[^'\"]+)['\"] con el asunto ['\"](?P<subject>[^'\"]+)['\"] "
    r"y con el mensaje ['\"](?P<body>[^'\"]+)['\"]",
    re.IGNORECASE,
)


def _send_via_gmail(subject: str, sender: str, recipient: str, body: str) -> None:
    # >>> Requires GMAIL_USER and GMAIL_APP_PASSWORD in the .env file.
    if not config.GMAIL_USER or not config.GMAIL_APP_PASSWORD:
        raise RuntimeError(
            "Gmail credentials are not configured. "
            "Set GMAIL_USER and GMAIL_APP_PASSWORD in your .env file."
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(config.GMAIL_USER, config.GMAIL_APP_PASSWORD)
        smtp.send_message(message)


def make_email_tool() -> Tool:
    def _run(instruction: str) -> str:
        match = _EMAIL_PATTERN.search(instruction)
        if not match:
            return (
                "Could not parse the email instruction. Use the format: "
                "envia un correo a 'addr@x.com' con el asunto 'SUBJECT' y "
                "con el mensaje 'BODY'."
            )
        try:
            _send_via_gmail(
                subject=match.group("subject").strip(),
                sender=config.GMAIL_USER,
                recipient=match.group("to").strip(),
                body=match.group("body").strip(),
            )
            return "Email sent successfully."
        except Exception as exc:  # noqa: BLE001
            return f"Failed to send email: {exc}"

    return Tool(
        name="send_email",
        func=_run,
        description=(
            "Send an email through Gmail. The input must be a sentence shaped "
            "like: \"envia un correo a 'address@example.com' con el asunto "
            "'SUBJECT' y con el mensaje 'BODY'\"."
        ),
    )


def build_default_toolset(index: RagIndex) -> List[Tool]:
    """Convenience: return the three tools wired together."""
    return [
        make_document_search_tool(index),
        make_web_search_tool(),
        make_email_tool(),
    ]
