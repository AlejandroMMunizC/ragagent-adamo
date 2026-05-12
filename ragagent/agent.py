"""Factory that assembles the LangChain agent."""

from __future__ import annotations

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI

from . import config
from .rag import RagIndex
from .tools import build_default_toolset


def build_agent(index: RagIndex):
    """Return a LangChain agent wired to the local RAG index and external tools."""
    llm = ChatOpenAI(
        api_key=config.require_openai_key(),
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )
    tools = build_default_toolset(index)

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
