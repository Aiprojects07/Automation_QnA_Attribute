"""
Search and Answer core utilities

This module centralizes configuration and helpers for:
- Anthropic client (Claude Sonnet 4)
- Prompt file path (path only; no prompt content here)
- Pinecone index name
- OpenAI embeddings for query vectorization

Environment variables used:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- OPENAI_EMBEDDING_MODEL (optional, defaults to text-embedding-3-large)
- PINECONE_INDEX_NAME (optional, defaults to lipstick-cluster)
"""

from __future__ import annotations

import os
import json
import textwrap
from typing import List, Optional, Sequence

try:
    # Optional: if python-dotenv is installed and a .env exists, load it.
    # Safe if missing; the rest of the code relies on environment variables.
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --- Anthropic (Claude) ---
from anthropic import Anthropic

# --- OpenAI (Embeddings) ---
from openai import OpenAI


# Model to use for Anthropic
ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

# Path-only constant to your vector search prompt file
PROMPT_PATH: str = \
    "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/data/QnA_prompt.json"

# Pinecone index name (env override supported)
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "qna-attributes")


def get_anthropic_client(api_key: Optional[str] = None) -> Anthropic:
    """Return an Anthropic client. If api_key is None, uses ANTHROPIC_API_KEY from env."""
    if api_key:
        return Anthropic(api_key=api_key)
    return Anthropic()


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client. If api_key is None, uses OPENAI_API_KEY from env."""
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def embed_query(text: str, model: Optional[str] = None) -> List[float]:
    """
    Convert a natural-language query into an embedding vector using OpenAI.

    Parameters:
    - text: the input query string
    - model: optional embedding model; if None, uses env OPENAI_EMBEDDING_MODEL
             or defaults to 'text-embedding-3-large'.

    Returns a list[float] embedding.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    model_name = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    client = get_openai_client()
    resp = client.embeddings.create(model=model_name, input=text)
    return resp.data[0].embedding


def _read_prompt_text(path: str) -> str:
    """Read prompt text from a file. If JSON with a 'prompt' key, return that; else raw text."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if path.endswith(".json"):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "prompt" in data:
                        return str(data["prompt"])  # type: ignore
                except Exception:
                    pass
            return content
    except FileNotFoundError:
        # Fallback minimal instruction if the prompt file is missing
        return (
            "You are Claude Sonnet 4. Answer the user's question using only the provided context. "
            "Cite facts found in context and do not invent details."
        )


def _format_context_blocks(contexts: Sequence[str], max_chars: int = 15000) -> str:
    """
    Join top-k context strings into a single block with separators, trimmed to max_chars.
    """
    safe_blocks = []
    running = 0
    for i, c in enumerate(contexts, 1):
        part = f"\n--- Context #{i} ---\n{c.strip()}\n"
        if running + len(part) > max_chars:
            remaining = max_chars - running
            if remaining > 0:
                part = part[:remaining]
                safe_blocks.append(part)
            break
        safe_blocks.append(part)
        running += len(part)
    return "".join(safe_blocks).strip()


def generate_answer_with_claude(
    query: str,
    contexts: Sequence[str],
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """
    Use Claude Sonnet 4 to generate an answer from top-k retrieved contexts.

    Parameters:
    - query: user query string
    - contexts: a sequence of text snippets retrieved from the vector DB (top-k)
    - max_tokens: generation cap
    - temperature: sampling temperature

    Returns: model's text response
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    system_text = _read_prompt_text(PROMPT_PATH)
    context_text = _format_context_blocks(contexts)

    user_payload = textwrap.dedent(
        f"""
        <Query>
        {query.strip()}
        </Query>

        <RetrievedContext>
        {context_text}
        </RetrievedContext>

        Please answer the query strictly based on the RetrievedContext. If the context
        is insufficient, say you don't have enough information.
        """
    ).strip()

    client = get_anthropic_client()
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_text,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_payload},
                ],
            }
        ],
    )

    # Extract text parts from response
    out_parts: List[str] = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "text":
            out_parts.append(getattr(block, "text", ""))
    return "".join(out_parts).strip()


__all__ = [
    "ANTHROPIC_MODEL",
    "PROMPT_PATH",
    "PINECONE_INDEX_NAME",
    "get_anthropic_client",
    "get_openai_client",
    "embed_query",
    "generate_answer_with_claude",
]
