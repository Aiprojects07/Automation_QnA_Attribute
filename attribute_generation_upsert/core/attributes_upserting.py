#!/usr/bin/env python3
"""
Upsert attributes data (one record per product) into Pinecone.

- Input: Excel produced by attribute extraction pipeline
  Default: /home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/output/attributes_output.xlsx

- Output: Vectors upserted into Pinecone in a separate namespace (default: "attributes") to avoid
  overriding existing QnA vectors.

Record format (example), excluding `source` and `attributes_keys` as requested:
{
  "id": "12345::attrs",
  "values": [/* embedding */],
  "metadata": {
    "sku": "12345",
    "product_name": "Milani Ludicrous Matte Lip Crayon 120 Can't Even",
    "brand": "Milani",
    "product_line": "Ludicrous Matte Lip Crayon",
    "shade": "120 Can't Even",
    "category": "Lipstick",
    "sub_category": "",
    "leaf_level_category": "",
    "language": "en",
    "attributes_json": "{...}",
    "chunk_index": 0,
    "total_chunks": 1,
    "content_len": 412,
    "parent_id": "12345::attrs"
  }
}

Content that gets embedded (and stored in metadata.content) follows the same identity-header pattern used in QnA ingestion:

[ SKU=12345; Brand=Milani; Product=Milani Ludicrous Matte Lip Crayon 120 Can't Even; Product Line=Ludicrous Matte Lip Crayon; Shade=120 Can't Even; Category=Lipstick ]

Attributes
- finish: Matte
- coverage: Full
- transfer_proof: true
- key_ingredients: Shea Butter; Vitamin E

ENV (set before running):
  PINECONE_API_KEY=...
  OPENAI_API_KEY=...
  PINECONE_INDEX_NAME=qna-attributes               # or your existing index
  PINECONE_ENVIRONMENT=us-east-1                  # region for serverless index
  PINECONE_NAMESPACE=default                      # namespace to use
  OPENAI_EMBEDDING_MODEL=text-embedding-3-large   # embedding model
  ATTRIBUTES_INPUT_XLSX=/path/to/attributes_output.xlsx
  BATCH_SIZE=100

Usage:
  python attributes_upserting.py
"""

import os
import re
import json
import logging
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

import pandas as pd

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI


# -------------------- utilities --------------------

def ensure_index(index_name: str, region: str, dimension: int) -> "Index":
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    names = {i["name"] for i in pc.list_indexes()}
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )
    return pc.Index(index_name)


def embed_texts(oai: OpenAI, model: str, texts: List[str], batch_size: int = 100) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = oai.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return out


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


def flatten_attributes_lines(attrs: Dict[str, Any]) -> List[str]:
    lines: List[str] = ["Attributes"]
    for k, v in attrs.items():
        if isinstance(v, list):
            joined = "; ".join([normalize_bool_str(x) for x in v])
            lines.append(f"- {k}: {joined}")
        elif isinstance(v, dict):
            # flatten one level
            inner = "; ".join([f"{ik}={normalize_bool_str(iv)}" for ik, iv in v.items()])
            lines.append(f"- {k}: {inner}")
        else:
            lines.append(f"- {k}: {normalize_bool_str(v)}")
    return lines


def build_identity_header(meta: Dict[str, str]) -> str:
    header_kv = []
    for key, label in [
        ("sku", "SKU"),
        ("brand", "Brand"),
        ("product_name", "Product"),
        ("product_line", "Product Line"),
        ("shade", "Shade"),
        ("category", "Category"),
        ("sub_category", "Sub Category"),
        ("leaf_level_category", "Leaf Level Category"),
    ]:
        val = meta.get(key)
        if val:
            header_kv.append(f"{label}={val}")
    return f"[ {'; '.join(header_kv)} ]" if header_kv else ""


def normalize_columns(row: pd.Series) -> Dict[str, str]:
    # Lowercase, alnum+underscore keys
    norm = {re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_"): ("" if pd.isna(v) else str(v).strip()) for k, v in row.items()}
    # Map common fields
    sku = norm.get("kult_sku_code") or norm.get("sku") or ""
    brand = norm.get("brand") or ""
    product_line = norm.get("product_name") or norm.get("product") or norm.get("product_title") or norm.get("line") or ""
    shade = norm.get("shade") or norm.get("shade_of_lipstick") or norm.get("color") or ""
    category = norm.get("category") or ""
    sub_category = norm.get("sub_category") or norm.get("sub_sub_category") or ""
    leaf_level_category = norm.get("leaf_level_category") or norm.get("sub_sub_category") or ""

    # Build display product_name (brand + product_line + shade)
    display_name = " ".join([x for x in [brand, product_line, shade] if x]).strip()

    # attributes_json field name handling
    attrs_json_s = norm.get("attributes_json", "")

    return {
        "sku": sku,
        "brand": brand,
        "product_line": product_line,
        "shade": shade,
        "category": category,
        "sub_category": sub_category,
        "leaf_level_category": leaf_level_category,
        "product_name": display_name,
        "attributes_json": attrs_json_s,
    }


# -------------------- main --------------------


def main():
    # Load env
    load_dotenv(override=True)


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('attributes_ingest.log')
        ]
    )
    logger = logging.getLogger(__name__)


    # Config strictly from environment
    input_xlsx = os.getenv(
        "ATTRIBUTES_INPUT_XLSX",
        "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/output/attributes_output.xlsx",
    )
    index_name = os.getenv("PINECONE_INDEX_NAME")
    namespace = os.getenv("PINECONE_NAMESPACE")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    # Fixed batch size defined here (not from environment)
    batch_size = 100


    # Validate required env vars
    missing: List[str] = []
    for key, val in [
        ("ATTRIBUTES_INPUT_XLSX", input_xlsx),
        ("PINECONE_INDEX_NAME", index_name),
        ("PINECONE_NAMESPACE", namespace),
        ("PINECONE_ENVIRONMENT", environment),
        ("OPENAI_EMBEDDING_MODEL", embedding_model),
        ("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY")),
        ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
    ]:
        if not val:
            missing.append(key)


    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing) +
            "\nPlease set them in your .env file."
        )


    logger.info(
        "Config: index=%s namespace=%s env=%s model=%s batch_size=%s",
        index_name, namespace, environment, embedding_model, batch_size
    )
    logger.info("Input: %s", input_xlsx)


    # Read Excel
    try:
        df = pd.read_excel(input_xlsx, engine="openpyxl")
    except Exception as e:
        logger.error("Failed reading Excel: %s", e)
        raise


    if df.empty:
        logger.warning("Input Excel is empty; nothing to upsert")
        return


    # Build records
    records: List[Dict[str, Any]] = []
    skipped = 0
    for _, row in df.iterrows():
        fields = normalize_columns(row)
        sku = fields["sku"].strip()
        if not sku:
            skipped += 1
            continue


        # Parse attributes_json
        attrs_obj: Dict[str, Any] = {}
        attrs_raw = fields.get("attributes_json", "").strip()
        if attrs_raw:
            try:
                attrs_obj = json.loads(attrs_raw)
                if not isinstance(attrs_obj, dict):
                    attrs_obj = {"value": attrs_obj}
            except Exception:
                # Store raw in an error wrapper but still proceed
                attrs_obj = {"_raw": attrs_raw, "_parse_error": True}
        else:
            # No attributes present, still upsert minimal record
            attrs_obj = {}


        # Build identity and content
        base_meta = {
            "sku": sku,
            "product_name": fields.get("product_name", ""),
            "brand": fields.get("brand", ""),
            "product_line": fields.get("product_line", ""),
            "shade": fields.get("shade", ""),
            "category": fields.get("category", ""),
            "sub_category": fields.get("sub_category", ""),
            "leaf_level_category": fields.get("leaf_level_category", ""),
            "language": "en",
        }
        identity_header = build_identity_header(base_meta)


        attr_lines = flatten_attributes_lines(attrs_obj) if attrs_obj else ["Attributes", "- (none)"]
        content_body = "\n".join(attr_lines)
        full_text = f"{identity_header}\n\n{content_body}" if identity_header else content_body
        full_text = re.sub(r"\s+", " ", full_text).strip()  # normalize spaces like QnA pipeline


        rec_id = f"{sku}::attrs"
        metadata = {
            **base_meta,
            "chunk_index": 0,
            "total_chunks": 1,
            "content_len": len(full_text),
            "parent_id": rec_id,
            "content": full_text,
        }
        records.append({
            "id": rec_id,
            "values": None,
            "metadata": metadata,
        })


    if not records:
        logger.warning("No valid records to upsert (skipped=%s)", skipped)
        return


    logger.info("Prepared %s records (skipped=%s)", len(records), skipped)


    # Embeddings
    try:
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        texts = [r["metadata"]["content"] for r in records]
        vectors = embed_texts(oai, embedding_model, texts, batch_size=batch_size)
        if not vectors:
            raise RuntimeError("No vectors produced; check inputs.")
        dim = len(vectors[0])
        for rec, vec in zip(records, vectors):
            rec["values"] = vec
    except Exception as e:
        logging.getLogger(__name__).error("Embedding generation failed: %s", e)
        raise


    # Upsert
    try:
        index = ensure_index(index_name, environment, dimension=dim)
        total = 0
        for i in range(0, len(records), batch_size):
            part = records[i:i+batch_size]
            index.upsert(vectors=part, namespace=namespace)
            total += len(part)
            logging.getLogger(__name__).info("Upserted batch %s: %s records", i // batch_size + 1, len(part))
        print(f"[ok] Upserted {total} attribute vectors to index='{index_name}', namespace='{namespace}'.")
    except Exception as e:
        logging.getLogger(__name__).error("Upsert failed: %s", e)
        raise



if __name__ == "__main__":
    main()