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
- PINECONE_NAMESPACE (optional, defaults to default)
- PINECONE_API_KEY
"""

from __future__ import annotations

import os
import json
import textwrap
import csv
import re
import pprint
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

# --- Pinecone (Vector DB) ---
from pinecone import Pinecone


# Model to use for Anthropic
ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

# Path-only constant to your vector search prompt file
PROMPT_PATH: str = \
    "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/data/QnA_prompt.json"

# Pinecone index name (env override supported)
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "qna-attributes")
# Pinecone namespace (env override supported)
PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "default")

# Retrieval size (centralized)
DEFAULT_TOP_K: int = 5

# CLI-independent defaults
DEFAULT_CSV_PATH: str = \
    "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/core/lipstick_list.csv"
DEFAULT_MAX_TOKENS: int = 15000
DEFAULT_TEMPERATURE: float = 0.2

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
            if not content:
                raise ValueError(f"Prompt file at '{path}' is empty.")
            if path.endswith(".json"):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "prompt" in data:
                        return str(data["prompt"])  # type: ignore
                except Exception:
                    pass
            return content
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Prompt file not found at '{path}'. Please create it or set PROMPT_PATH."
        ) from e


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
    # Pass contexts directly without extra labels or truncation
    context_text = "\n\n".join((c or "").strip() for c in contexts)

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


def get_pinecone_index(index_name: Optional[str] = None):
    """Return a Pinecone Index handle using PINECONE_API_KEY from env.

    Parameters:
    - index_name: optional override; defaults to PINECONE_INDEX_NAME
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment")
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name or PINECONE_INDEX_NAME)


def search_contexts(
    query: str,
    namespace: Optional[str] = None,
    metadata_filter: Optional[dict] = None,
) -> List[str]:
    """Search Pinecone for top_k contexts, with optional metadata filtering.

    - query: natural language query string
    - namespace: optional Pinecone namespace; defaults to PINECONE_NAMESPACE
    - metadata_filter: optional dict for Pinecone metadata filter (e.g., {"sku": "...", "brand": {"$in": ["..."]}})

    Returns a list of context strings (from match.metadata["content"]).
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    vec = embed_query(query)
    index = get_pinecone_index()
    ns = namespace or PINECONE_NAMESPACE

    debug = os.getenv("SEARCH_DEBUG", "0") == "1"
    if debug:
        print("[search] index=", PINECONE_INDEX_NAME)
        print("[search] namespace=", ns)
        print("[search] top_k=", DEFAULT_TOP_K)
        print("[search] filter=", pprint.pformat(metadata_filter))
        # Describe index stats to see if any vectors match the filter
        try:
            stats = index.describe_index_stats(filter=metadata_filter) if metadata_filter else index.describe_index_stats()
            # stats.namespaces is a dict: { namespace: {"vectorCount": N, ...}, ... }
            ns_stats = getattr(stats, "namespaces", None) or {}
            ns_info = ns_stats.get(ns) or {}
            print("[search] namespace vectorCount=", ns_info.get("vectorCount"))
            if metadata_filter:
                # When a filter is provided, Pinecone narrows stats accordingly.
                print("[search] stats (filtered) namespaces keys=", list(ns_stats.keys()))
        except Exception as e:
            print("[search][warn] describe_index_stats failed:", e)

    resp = index.query(
        vector=vec,
        top_k=DEFAULT_TOP_K,
        namespace=ns,
        filter=metadata_filter,  # use provided filter as-is (can be None)
        include_metadata=True,
    )
    contexts: List[str] = []
    for m in getattr(resp, "matches", []) or []:
        md = getattr(m, "metadata", None) or {}
        c = md.get("content") if isinstance(md, dict) else None
        if isinstance(c, str) and c.strip():
            contexts.append(c.strip())
    if debug:
        print(f"[search] matches={len(contexts)}")
    return contexts


def _normalize_filter_value(val):
    """Normalize a filter value for Pinecone filter: str -> exact match, list/tuple -> $in."""
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        seq = [v for v in val if v is not None and str(v).strip()]
        return {"$in": seq} if seq else None
    s = str(val).strip()
    return s if s else None


def build_metadata_filter(
    *,
    sku: Optional[str | List[str]] = None,
    category: Optional[str | List[str]] = None,
) -> dict:
    """Build a Pinecone-compatible metadata filter using selected fields only.

    Allowed fields: sku, category.
    Pass strings for exact match or lists for $in matching.
    """
    filt: dict = {}
    for key, val in [
        ("sku", sku),
        ("category", category),
    ]:
        norm = _normalize_filter_value(val)
        if norm is not None:
            filt[key] = norm

    return filt


def build_identity_query(
    *,
    sku: str = "",
    brand: str = "",
    product_name: str = "",
    product_line: str = "",
    shade: str = "",
    category: str = "",
    sub_category: str = "",
    leaf_level_category: str = "",
) -> str:
    """Create a compact identity header string mirroring the upserted metadata header.

    This improves retrieval because the upsert pipeline prefixed each record's content
    with an identity header like:
      [ SKU=...; Brand=...; Product=...; Product Line=...; Shade=...; Category=...; ... ]

    We build a similar string here to maximize embedding similarity.
    """
    parts = []
    if sku: parts.append(f"SKU={sku}")
    if brand: parts.append(f"Brand={brand}")
    if product_name: parts.append(f"Product={product_name}")
    if product_line: parts.append(f"Product Line={product_line}")
    if shade: parts.append(f"Shade={shade}")
    if category: parts.append(f"Category={category}")
    if sub_category: parts.append(f"Sub Category={sub_category}")
    if leaf_level_category: parts.append(f"Leaf Level Category={leaf_level_category}")
    header = f"[ {'; '.join(parts)} ]" if parts else ""
    return header or "product"


def answer_with_filters(
    *,
    query: str,
    namespace: Optional[str] = None,
    sku: Optional[str | List[str]] = None,
    category: Optional[str | List[str]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """High-level helper: search Pinecone with metadata filters and answer via Claude.

    Provide any of sku/category to restrict the search within the same index/namespace.
    """
    meta_filter = build_metadata_filter(
        sku=sku,
        category=category,
    )

    contexts = search_contexts(
        query=query,
        namespace=namespace,
        metadata_filter=meta_filter or None,
    )
    if not contexts:
        raise RuntimeError("No contexts found for the given filters; adjust filters or data.")

    return generate_answer_with_claude(
        query=query,
        contexts=contexts,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _cli_main(argv: Optional[Sequence[str]] = None) -> int:
    import sys
    # Use module-level defaults instead of CLI args
    csv_path = DEFAULT_CSV_PATH
    namespace = PINECONE_NAMESPACE
    max_tokens = DEFAULT_MAX_TOKENS
    temperature = DEFAULT_TEMPERATURE

    # Load product details from CSV
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"[error] Failed to read CSV at {csv_path}: {e}", file=sys.stderr)
        return 2

    if not rows:
        print("[error] CSV contains no rows", file=sys.stderr)
        return 2

    # Select the first row automatically (no CLI selection)
    row = rows[0]

    # Normalize headers to snake_case: lower, non-alnum -> '_', trim underscores
    norm_row = { re.sub(r"[^a-z0-9]+", "_", (k or "").lower()).strip("_"): (v or "").strip() for k, v in row.items() }

    # Map normalized fields to our canonical names
    sku = norm_row.get("kult_sku_code") or norm_row.get("sku") or ""
    brand = norm_row.get("brand") or ""
    # CSV provides the line name under Product_name; treat that as product_line
    product_line = (
        norm_row.get("product_name")
        or norm_row.get("product")
        or norm_row.get("product_title")
        or norm_row.get("line")
        or ""
    )
    # Build a full canonical product_name = "Brand Product_line Shade"
    product_name = (f"{brand} {product_line} {norm_row.get('shade') or ''}").strip()
    shade = (
        norm_row.get("shade")
        or norm_row.get("shade_of_lipstick")
        or norm_row.get("color")
        or ""
    )
    category = norm_row.get("category") or norm_row.get("product_category") or norm_row.get("category_name") or ""
    sub_category = norm_row.get("sub_category") or norm_row.get("subcategory") or norm_row.get("sub_category_name") or ""
    leaf_level_category = (
        norm_row.get("sub_sub_category")
        or norm_row.get("leaf_level_category")
        or norm_row.get("leaflevelcategory")
        or norm_row.get("leaf_level_cat")
        or ""
    )

    meta = build_metadata_filter(
        sku=sku or None,
        category=category or None,
    )

    # Query should be only the product_name (semantic search), filtering narrows scope
    identity_query = product_name

    contexts = search_contexts(
        query=identity_query,
        namespace=namespace,
        metadata_filter=meta or None,
    )
    if not contexts:
        print("[error] No contexts found for the given filters.", file=sys.stderr)
        return 2

    answer = generate_answer_with_claude(
        # Query text: product name only; prompt defines extraction behavior
        query=identity_query,
        contexts=contexts,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
