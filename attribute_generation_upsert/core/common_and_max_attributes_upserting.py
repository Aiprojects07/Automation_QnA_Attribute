#!/usr/bin/env python3
"""
Upsert 'common/max attributes' JSON outputs into Pinecone.

- Input files: one JSON per SKU produced by
  attribute_generation_upsert/core/common_and_max_attributes_generation.py
  Default input directory:
    /home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/maximum_and_common_Attributes_per_product

- Pinecone:
  Index name: common-attributes   (created automatically if it doesn't exist)
  Namespace: from env PINECONE_NAMESPACE (default: "default")

Environment required:
  PINECONE_API_KEY=...
  OPENAI_API_KEY=...
  PINECONE_NAMESPACE=default
  OPENAI_EMBEDDING_MODEL=text-embedding-3-large    (default)
  PINECONE_ENVIRONMENT=us-east-1                   (serverless region)

Usage:
  python common_and_max_attributes_upserting.py
"""

import os
import re
import json
import glob
import logging
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI


# -------------------- configuration --------------------

DEFAULT_INPUT_DIR = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/maximum_and_common_Attributes_per_product"
INDEX_NAME = "common-attributes"  # requested name

# -------------------- logging --------------------

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# -------------------- helpers --------------------

def ensure_index(pc: Pinecone, index_name: str, region: str, dimension: int) -> "Index":
    names = {i["name"] for i in pc.list_indexes()}
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )
    return pc.Index(index_name)


def ensure_str(x: Any) -> str:
    return "" if x is None else str(x)


def normalize_bool_str(val: Any) -> str:
    # Represent booleans as lowercase true/false for consistent text
    if isinstance(val, bool):
        return "true" if val else "false"
    s = ensure_str(val).strip()
    if s.lower() in ("true", "false"):
        return s.lower()
    return s


def flatten_dict_to_lines(obj: Any, prefix: str = "") -> List[str]:
    """
    Flattens nested JSON into human-readable lines:
      - scalar -> "prefix: value"
      - list   -> "prefix: v1; v2; v3"
      - dict   -> one-level 'k=v' pairs if shallow, or recurse with dotted keys
    """
    lines: List[str] = []
    if isinstance(obj, dict):
        # If shallow and non-nested, compress to "k=v; k2=v2"
        shallow_pairs = []
        nested_keys = False
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                nested_keys = True
                break
            shallow_pairs.append(f"{k}={normalize_bool_str(v)}")

        if shallow_pairs and not nested_keys and prefix:
            lines.append(f"{prefix}: " + "; ".join(shallow_pairs))
        else:
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                lines.extend(flatten_dict_to_lines(v, new_prefix))
    elif isinstance(obj, list):
        flat_vals = "; ".join([normalize_bool_str(x) for x in obj])
        if prefix:
            lines.append(f"{prefix}: {flat_vals}")
        else:
            lines.append(flat_vals)
    else:
        if prefix:
            lines.append(f"{prefix}: {normalize_bool_str(obj)}")
        else:
            lines.append(normalize_bool_str(obj))
    return lines


def build_text_from_attributes(attrs: Dict[str, Any]) -> str:
    """
    Build an embedding-ready text from the attributes JSON by flattening to readable lines.
    """
    lines = flatten_dict_to_lines(attrs)
    # Normalize spacing a bit
    text = "\n".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_json_file(fp: str) -> Dict[str, Any]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            return data
        # If it's not a dict, wrap it
        return {"value": data}

# -------------------- main --------------------

def main():
    load_dotenv(override=True)
    logger = setup_logging()

    input_dir = os.getenv(
        "COMMON_ATTR_INPUT_DIR",
        DEFAULT_INPUT_DIR
    )
    namespace = os.getenv("PINECONE_NAMESPACE", "default")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    serverless_region = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    # Validate secrets early
    for key in ["PINECONE_API_KEY", "OPENAI_API_KEY"]:
        if not os.getenv(key):
            raise RuntimeError(f"Missing required env: {key}")

    logger.info("Input directory: %s", input_dir)
    logger.info("Pinecone index: %s (namespace=%s, region=%s)", INDEX_NAME, namespace, serverless_region)
    logger.info("Embedding model: %s", embedding_model)

    # Collect JSON files
    pattern = os.path.join(input_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No JSON files found in %s", input_dir)
        return

    # Build records (text + metadata), then embed
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    ids: List[str] = []
    for fp in files:
        try:
            attrs = read_json_file(fp)
        except Exception as e:
            logger.error("Failed to parse JSON %s: %s", fp, e)
            continue

        # SKU from filename
        base = os.path.basename(fp)
        sku = os.path.splitext(base)[0].strip()

        text = build_text_from_attributes(attrs)
        if not text:
            # Still upsert minimal text to keep a record
            text = "(empty attributes)"

        rec_id = f"{sku}::common_attrs"
        meta = {
            "sku": sku,
            "content": text,
            "content_len": len(text),
            "parent_id": rec_id,
            "type": "common_attributes",
        }

        ids.append(rec_id)
        texts.append(text)
        metas.append(meta)

    if not ids:
        logger.warning("No valid records to upsert.")
        return

    # Create embeddings
    try:
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        vectors: List[List[float]] = []
        # Chunk for embedding to stay below token limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            resp = oai.embeddings.create(model=embedding_model, input=chunk)
            vectors.extend([d.embedding for d in resp.data])
        if not vectors:
            raise RuntimeError("No vectors produced.")
        dim = len(vectors[0])
        logger.info("Embedding dimension resolved: %s", dim)
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        raise

    # Ensure Pinecone index and upsert
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = ensure_index(pc, INDEX_NAME, serverless_region, dimension=dim)

        # Prepare upsert payloads
        payload = []
        for _id, vec, meta in zip(ids, vectors, metas):
            payload.append({
                "id": _id,
                "values": vec,
                "metadata": meta,
            })

        total = 0
        upsert_batch = 100
        for i in range(0, len(payload), upsert_batch):
            part = payload[i:i+upsert_batch]
            index.upsert(vectors=part, namespace=namespace)
            total += len(part)
            logger.info("Upserted batch %s: %s records", i // upsert_batch + 1, len(part))

        logger.info("[ok] Upserted %s vectors to index='%s', namespace='%s'.", total, INDEX_NAME, namespace)
        print(f"[ok] Upserted {total} attribute vectors to index='{INDEX_NAME}', namespace='{namespace}'.")

    except Exception as e:
        logger.error("Upsert failed: %s", e)
        raise


if __name__ == "__main__":
    main()