# In this file there is need to change only report json file path
#!/usr/bin/env python3
"""
Ingest a Markdown-derived Q&A report JSON into Pinecone (full coverage).

Target JSON shape (from your converter):
{
  "title": "...",
  "source_file": "nars-dolce-vita-report.md",
  "extracted_at_utc": "...",
  "product": {
    "brand": "NARS",
    "product_line": "Air Matte Lip Color",
    "shade": "Dolce Vita",
    "full_name": "NARS Air Matte Lip Color - Dolce Vita: Complete Q&A Diagnostic Report"
  },
  "sections": [
    {
      "title": "Section 1: ...",
      "qas": [
        {"q": "...", "a": "...", "why": "...", "solution": "..."},
        ...
      ]
    },
    {
      "title": "Bottom Line",
      "content": "..."
    },
    {
      "title": "Quick Reference Snapshot",
      "snapshot": { "Feature": "Value", ... }
    }
  ]
}

What this script does:
- Creates deterministic, idempotent Pinecone IDs
- Chunks:
    * one meta chunk for the product
    * one chunk per Section.content
    * one chunk per Section.snapshot (flattened)
    * one chunk per QA (q+a+why+solution)
    * one raw_json_full catch-all chunk
- Embeds with OpenAI (text-embedding-3-large by default)
- Auto-creates the Pinecone index (serverless) if missing

ENV (set before running):
  PINECONE_API_KEY=...
  OPENAI_API_KEY=...
  PINECONE_INDEX_NAME=products-general
  PINECONE_NAMESPACE=reports
  PINECONE_ENVIRONMENT=us-east-1
  OPENAI_EMBEDDING_MODEL=text-embedding-3-large
  BATCH_SIZE=100
  FILE_PATH=data/nars-dolce-vita-report-2.json

Usage:
  python ingest_report_json_to_pinecone.py
"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

# pip install pinecone-client openai python-dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Optional: automatic JSON repair for slightly malformed JSON outputs
try:
    from json_repair import repair_json as _repair_json  # type: ignore
except Exception:
    _repair_json = None  # type: ignore


def try_repair_json(s: str) -> Optional[str]:
    """Attempt to repair malformed JSON text using json-repair if available.
    Returns a repaired JSON string that validates with json.loads, else None.
    """
    if not s:
        return None
    if _repair_json is None:
        return None
    try:
        repaired = _repair_json(s)
        # Validate repair result
        json.loads(repaired)
        return repaired
    except Exception:
        return None


# -------------------- utilities --------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def slugify(t: str) -> str:
    t = (t or "").lower()
    # keep only a-z, 0-9, spaces and hyphens during normalization
    t = re.sub(r"[^a-z0-9\s-]+", "", t)
    # convert whitespace and slashes to single hyphen
    t = re.sub(r"[\s/]+", "-", t)
    # collapse repeated hyphens and trim
    t = re.sub(r"-+", "-", t).strip("-")
    return t or "unknown"

def stable_id(*parts: str) -> str:
    key = "|".join(parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:28]

def ensure_str(x: Any) -> str:
    return "" if x is None else str(x)

def flatten_snapshot(snapshot: Dict[str, Any]) -> str:
    lines = []
    for k, v in snapshot.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)

def extract_first_json_object(text: str) -> Optional[str]:
    """Heuristically extract the first top-level JSON object from a string.
    Returns the JSON substring if found, else None.
    """
    s = (text or "").strip()
    # Fast path
    try:
        json.loads(s)
        return s
    except Exception:
        # Try repairing the whole text in case it's a single object with minor issues
        repaired_full = try_repair_json(s)
        if repaired_full is not None:
            return repaired_full
        pass
    start = s.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    string_char = ''
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == string_char:
                in_string = False
        else:
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        # Attempt repair for this candidate before continuing scan
                        repaired = try_repair_json(candidate)
                        if repaired is not None:
                            return repaired
                        # keep scanning
                        pass
    # fallback: from first '{' to last '}'
    last_close = s.rfind('}')
    if last_close != -1 and last_close > start:
        candidate = s[start:last_close+1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            # Attempt repair for the wide candidate slice
            repaired = try_repair_json(candidate)
            if repaired is not None:
                return repaired
            return None


# -------------------- Pinecone + embeddings --------------------

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

def split_large_text(text: str, max_tokens: int = 8000) -> List[str]:
    """Split large text into chunks that fit within token limits (rough approximation: 1 token ≈ 4 chars)"""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a sentence or paragraph boundary
        chunk = text[start:end]
        last_period = chunk.rfind('.')
        last_newline = chunk.rfind('\n')
        
        if last_period > len(chunk) * 0.8:  # If period is in last 20%
            end = start + last_period + 1
        elif last_newline > len(chunk) * 0.8:  # If newline is in last 20%
            end = start + last_newline + 1
        
        chunks.append(text[start:end])
        start = end
    
    return chunks

def embed_texts(oai: OpenAI, model: str, texts: List[str], batch_size: int = 100) -> List[List[float]]:
    out: List[List[float]] = []
    
    # Embed texts as-is (text splitting is handled at record level)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = oai.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    
    return out


# -------------------- records builder --------------------

def build_records_from_report(data: Dict[str, Any], source_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Identity
    product = data.get("product") or {}
    # Fetch full canonical product name exclusively from report JSON
    product_name = product.get("full_name") or ""
    brand = product.get("brand") or ""
    product_line = product.get("product_line") or ""
    shade = product.get("shade") or ""
    category = product.get("category") or ""
    sub_category = product.get("sub_category") or ""
    leaf_level_category = product.get("leaf_level_category") or ""
    product_type = product.get("productType") or product.get("product_type") or ""
    # Enforce mandatory fields
    if not product_name:
        raise ValueError("product.full_name is required in the report JSON to build product identity.")
    if not shade:
        raise ValueError("product.shade is required in the report JSON to build product identity (Brand Product Line - Shade).")
    # Require the unique SKU from the report JSON as the product_id
    sku = product.get("sku") or ""
    if not sku:
        raise ValueError("product.sku is required in the report JSON to build product identity (must be unique per product).")
    product_id = sku

    base_meta = {
        "sku": sku,
        "product_name": product_name,
        "brand": brand,
        "product_line": product_line,
        "shade": shade,
        "category": category,       # family name for this JSON type
        "sub_category": sub_category,
        "leaf_level_category": leaf_level_category,
        "product_type": product_type,
        "language": "en",
    }

    # Build one record per section (title + optional content + QAs)
    records: List[Dict[str, Any]] = []

    # Add a Product Overview record so product-level fields (including keyFeatures) are indexed
    try:
        overview_parts: List[str] = []
        title_overview = "Product Overview"
        # Gather product-level narrative fields
        shade_desc = ensure_str(product.get("shade_description")).strip()
        finish_desc = ensure_str(product.get("finish_description")).strip()
        description = ensure_str(product.get("description")).strip()
        key_features = product.get("keyFeatures") or product.get("key_features") or []
        top_note = ensure_str(data.get("Note") or data.get("note")).strip()
        # Compose overview content
        header = f"# {title_overview}"
        overview_parts.append(header)
        if product_type:
            overview_parts.append(f"Product Type: {product_type}")
        if description:
            overview_parts.append(f"Description: {description}")
        if finish_desc:
            overview_parts.append(f"Finish: {finish_desc}")
        if shade_desc:
            overview_parts.append(f"Shade Insight: {shade_desc}")
        if isinstance(key_features, list) and key_features:
            bullet_lines = [f"- {ensure_str(kf).strip()}" for kf in key_features if ensure_str(kf).strip()]
            if bullet_lines:
                overview_parts.append("Key Features:\n" + "\n".join(bullet_lines))
        if top_note:
            overview_parts.append(f"Note: {top_note}")
        overview_text = "\n\n".join([p for p in overview_parts if p])
        if overview_text.strip():
            record_overview = {
                "id": f"{sku}::product",
                "values": None,
                "metadata": {
                    **base_meta,
                    "section_index": -1,
                    "section_title": title_overview,
                    "content": overview_text,
                }
            }
            records.append(record_overview)
    except Exception:
        # Non-fatal: proceed even if overview build fails
        pass

    sections = data.get("sections") or []
    for s_idx, sec in enumerate(sections):
        section_parts: List[str] = []
        title = ensure_str(sec.get("title")).strip()
        if title:
            section_parts.append(f"# {title}")
        if sec.get("content"):
            content = sec.get("content")
            # Flatten dict-style content (e.g., {"Ingredients": "...", "Safety": "..."})
            if isinstance(content, dict):
                for k, v in content.items():
                    key_hdr = ensure_str(k).strip()
                    if isinstance(v, list):
                        vals = [ensure_str(x).strip() for x in v if ensure_str(x).strip()]
                        if vals:
                            section_parts.append(f"{key_hdr}:\n" + "\n".join([f"- {x}" for x in vals]))
                    else:
                        val_txt = ensure_str(v).strip()
                        if val_txt:
                            section_parts.append(f"{key_hdr}: {val_txt}")
            else:
                section_parts.append(ensure_str(content).strip())

        # Include PROS and CONS arrays when present
        try:
            # Helper for case-insensitive key lookup
            def _get_ci(d, keys):
                if not isinstance(d, dict):
                    return None
                lower_map = {str(k).strip().lower(): v for k, v in d.items()}
                for name in keys:
                    v = lower_map.get(name.lower())
                    if v is not None:
                        return v
                return None

            # Try at section root (case-insensitive)
            pros = _get_ci(sec, ["PROS", "pros"]) or []
            cons = _get_ci(sec, ["CONS", "cons"]) or []

            # Fallback: inside section.content if it's a dict
            content_obj = sec.get("content")
            if (not pros or not isinstance(pros, list)) and isinstance(content_obj, dict):
                pros = _get_ci(content_obj, ["PROS", "pros"]) or pros
            if (not cons or not isinstance(cons, list)) and isinstance(content_obj, dict):
                cons = _get_ci(content_obj, ["CONS", "cons"]) or cons

            # If string instead of list, split into lines
            if isinstance(pros, str):
                pros = [p.strip().lstrip("-+•✓ ") for p in pros.splitlines() if p.strip()]
            if isinstance(cons, str):
                cons = [c.strip().lstrip("-+•✗x ") for c in cons.splitlines() if c.strip()]

            # Debug: log what we found for this section
            if title and ("pros" in title.lower() or "cons" in title.lower()):
                print(f"DEBUG [{sku}]: Section '{title}' - pros={len(pros) if isinstance(pros, list) else 'N/A'}, cons={len(cons) if isinstance(cons, list) else 'N/A'}")

            if isinstance(pros, list) and pros:
                pros_lines = [f"+ {ensure_str(p).strip()}" for p in pros if ensure_str(p).strip()]
                if pros_lines:
                    section_parts.append("PROS:\n" + "\n".join(pros_lines))
            if isinstance(cons, list) and cons:
                cons_lines = [f"- {ensure_str(c).strip()}" for c in cons if ensure_str(c).strip()]
                if cons_lines:
                    section_parts.append("CONS:\n" + "\n".join(cons_lines))
        except Exception:
            # best-effort; continue if formatting fails
            pass

        # Include USER_CONSENSUS_SUMMARY when present
        try:
            consensus = ensure_str(sec.get("USER_CONSENSUS_SUMMARY") or sec.get("user_consensus_summary")).strip()
            if consensus:
                section_parts.append(f"USER CONSENSUS: {consensus}")
        except Exception:
            pass

        for q_idx, qa in enumerate(sec.get("qas") or []):
            q = ensure_str(qa.get("q")).strip()
            a = ensure_str(qa.get("a")).strip()
            why = ensure_str(qa.get("why")).strip()
            sol = ensure_str(qa.get("solution")).strip()
            # New: include per-QA confidence score/text when present
            conf = ensure_str(qa.get("CONFIDENCE") or qa.get("confidence")).strip()
            qa_block = [
                f"Q{q_idx+1}: {q}" if q else "",
                f"A: {a}" if a else "",
                f"WHY: {why}" if why else "",
                f"SOLUTION: {sol}" if sol else "",
                f"CONFIDENCE: {conf}" if conf else "",
            ]
            section_parts.append("\n".join([p for p in qa_block if p]))

        section_text = "\n\n".join([p for p in section_parts if p])

        # Unique id per section under the SKU
        r_id = f"{sku}::sec-{s_idx+1}"
        record = {
            "id": r_id,
            "values": None,
            "metadata": {
                **base_meta,
                "section_index": s_idx,
                "section_title": title,
                "content": section_text,
            }
        }
        records.append(record)

    return records, base_meta


# ---- Metadata size helpers (40KB Pinecone limit) ----

def _meta_size_bytes(m: Dict[str, Any]) -> int:
    return len(json.dumps(m, ensure_ascii=False).encode("utf-8"))


def ensure_metadata_limit(meta: Dict[str, Any], limit_bytes: int = 40 * 1024, logger: Optional[logging.Logger] = None) -> None:
    """Ensure serialized metadata (JSON) stays under Pinecone's per-vector 40KB limit by trimming
    only the 'content' field. Mutates meta in-place. Uses binary search for efficiency.
    """
    size = _meta_size_bytes(meta)
    if size <= limit_bytes:
        return

    original = meta.get("content", "")
    text = original if isinstance(original, str) else str(original)
    lo, hi = 0, len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        meta["content"] = text[:mid]
        sz = _meta_size_bytes(meta)
        if sz <= limit_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    meta["content"] = text[:best]
    final_sz = _meta_size_bytes(meta)
    if logger:
        logger.warning(
            "Metadata truncated to %s bytes (limit=%s, original=%s bytes, kept_chars=%s of %s)",
            final_sz, limit_bytes, size, best, len(text)
        )


def split_record_into_chunks(
    base_meta: Dict[str, Any],
    base_id: str,
    content_text: str,
    limit_bytes: int = 40 * 1024,
    overlap_chars: int = 256,
    identity_header: str = "",
) -> List[Dict[str, Any]]:
    """Split a record with large content into multiple records that each fit under metadata size limit.
    Returns a list of records with ids suffixed by '-chunk-{k}'.
    Chunks will include an overlap (prefix) of the last `overlap_chars` characters from the previous chunk
    to preserve continuity for retrieval.
    """
    chunks: List[Dict[str, Any]] = []
    text = content_text or ""

    # If an identity header is provided, and the content already starts with it, strip it out of the base text
    # so we can re-prepend it to EVERY chunk for consistent identity. This avoids duplicating headers in the 1st chunk.
    if identity_header:
        hdr = identity_header.strip()
        if hdr and text.startswith(hdr):
            # Remove the header and any immediate whitespace/newlines following
            text = text[len(hdr):].lstrip()

    # This header will be added to each piece before overlap/body for both sizing and emission
    header_prefix = (identity_header.strip() + "\n\n") if identity_header.strip() else ""

    # Compute overhead without content
    meta_no_content = {k: v for k, v in base_meta.items() if k != "content"}
    # We'll binary-search a slice length that fits when combined with overhead
    idx = 0
    chunk_num = 0
    n = len(text)
    # Guard: at least leave room for minimal content
    min_room = _meta_size_bytes({**meta_no_content, "content": ""})
    if min_room >= limit_bytes:
        # Overhead alone exceeds limit; drop non-essential fields (rare). Keep required identity and section keys.
        trimmed = {k: meta_no_content.get(k) for k in [
            "sku", "brand", "product_name", "product_line", "shade", "category", "sub_category", "leaf_level_category",
            "product_type", "language", "section_index", "section_title", "parent_id", "chunk_index", "total_chunks"
        ] if k in meta_no_content}
        meta_no_content = trimmed
        # If still too big, it will be caught by ensure_metadata_limit below
    while idx < n:
        # Determine overlap prefix for this chunk
        prefix = ""
        if chunk_num > 0 and overlap_chars > 0:
            start = max(idx - overlap_chars, 0)
            prefix = text[start:idx]

        lo, hi = 1, n - idx
        best_len = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_piece = header_prefix + prefix + text[idx:idx + mid]
            candidate_meta = {**meta_no_content, "content": candidate_piece}
            sz = _meta_size_bytes(candidate_meta)
            if sz <= limit_bytes:
                best_len = mid
                lo = mid + 1
            else:
                hi = mid - 1
        # If even with overlap the candidate doesn't fit (best_len may end up < 1), drop overlap for this chunk and retry sizing once
        if best_len < 1:
            prefix = ""
            lo, hi = 1, n - idx
            best_len = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate_piece = header_prefix + text[idx:idx + mid]
                candidate_meta = {**meta_no_content, "content": candidate_piece}
                sz = _meta_size_bytes(candidate_meta)
                if sz <= limit_bytes:
                    best_len = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

        # emit chunk
        piece = header_prefix + prefix + text[idx:idx + best_len]
        chunk_num += 1
        rec = {
            "id": f"{base_id}-chunk-{chunk_num}",
            "values": None,
            "metadata": {**meta_no_content, "content": piece}
        }
        # Safety ensure
        ensure_metadata_limit(rec["metadata"], limit_bytes=limit_bytes)
        chunks.append(rec)
        idx += best_len

    # annotate chunk counts
    total = len(chunks)
    for i, rec in enumerate(chunks):
        rec["metadata"]["chunk_index"] = i
        rec["metadata"]["total_chunks"] = total
    return chunks


# -------------------- main --------------------

def main():
    # Load environment variables from .env file with override
    load_dotenv(override=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ingest_reports.log')
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting report ingestion process")

    # Configuration from environment variables only
    file_path = os.getenv("FILE_PATH", "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/output/0sosyijthmprqf097stu7-milani-ludicrous-matte-lip-crayon-120-cant-even.json")
    index_name = os.getenv("PINECONE_INDEX_NAME", "qna-attributes")
    namespace = os.getenv("PINECONE_NAMESPACE", "default")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    batch_size = int(os.getenv("BATCH_SIZE", "100"))

    logger.info(f"Configuration: index={index_name}, namespace={namespace}, environment={environment}")
    logger.info(f"Embedding model: {embedding_model}, batch_size: {batch_size}")
    logger.info(f"File path: {file_path}")

    # env checks
    if "PINECONE_API_KEY" not in os.environ:
        logger.error("PINECONE_API_KEY not set")
        raise RuntimeError("PINECONE_API_KEY not set")
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not set")
        raise RuntimeError("OPENAI_API_KEY not set")

    logger.info("Environment variables validated successfully")

    # load JSON
    try:
        logger.info(f"Loading JSON data from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # Lenient parsing: ignore any preface/prose before the JSON object
        candidate = extract_first_json_object(raw_text)
        if candidate is None:
            # Fallback: try to find JSON by looking for first '{' and last '}'
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidate = raw_text[start_idx:end_idx + 1]
                logger.warning("JSON extraction function failed, using simple brace extraction")
            else:
                logger.error(f"Could not find valid JSON structure in file. First 200 chars: {raw_text[:200]}")
                candidate = raw_text
        try:
            data = json.loads(candidate)
        except Exception as e_strict:
            logger.warning(f"Strict JSON parse failed: {e_strict}. Attempting repair on candidate text.")
            repaired = try_repair_json(candidate)
            if repaired is None:
                # Final fallback: try repairing the entire raw_text span
                repaired = try_repair_json(raw_text)
            if repaired is None:
                # Give up with clear context
                raise
            data = json.loads(repaired)
            logger.info("JSON data repaired and loaded successfully (lenient parse)")
        else:
            logger.info("JSON data loaded successfully (lenient parse)")
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        logger.error(f"File preview (first 500 chars): {raw_text[:50] if 'raw_text' in locals() else 'N/A'}")
        raise

    # build records
    try:
        logger.info("Building records from report JSON")
        records, base_meta = build_records_from_report(data, source_name=os.path.basename(file_path))
        logger.info(f"Built {len(records)} records for sku='{base_meta['sku']}' product_name='{base_meta['product_name']}'")
        # Identity check print (visual confirmation before upsert)
        logger.info("IDENTITY | product_name='%s' sku='%s' brand='%s' line='%s' shade='%s'",
                    base_meta.get("product_name"), base_meta.get("sku"),
                    base_meta.get("brand"), base_meta.get("product_line"), base_meta.get("shade"))
        print(f"[info] Built {len(records)} records for sku='{base_meta['sku']}' product_name='{base_meta['product_name']}'.")
    except Exception as e:
        logger.error(f"Failed to build records: {e}")
        raise

    # embed
    try:
        logger.info("Initializing OpenAI client and generating embeddings")
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        texts = [r["metadata"]["content"] for r in records]
        logger.info(f"Generating embeddings for {len(texts)} text chunks with batch size {batch_size}")

        # Since we already build per-section records, skip additional text splitting.
        # Normalize content and annotate basic chunk metadata (single chunk per section).
        final_records = []
        for record in records:
            original_text = record["metadata"]["content"]
            cleaned_text = re.sub(r"\s+", " ", original_text).strip()
            single_rec = record.copy()
            single_rec["metadata"] = record["metadata"].copy()
            # Build a concise identity header from available metadata fields to help disambiguate similar content
            _m = single_rec["metadata"]
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
                ("product_type", "Product Type"),
            ]:
                val = _m.get(key)
                if val:
                    header_kv.append(f"{label}={val}")
            identity_header = f"[ {'; '.join(header_kv)} ]" if header_kv else ""
            prefixed_text = f"{identity_header}\n\n{cleaned_text}" if identity_header else cleaned_text

            # Prepare base metadata and id
            base_id = single_rec["id"]
            base_meta = single_rec["metadata"].copy()
            base_meta["content"] = prefixed_text
            base_meta["chunk_index"] = 0
            base_meta["total_chunks"] = 1
            base_meta["parent_id"] = record["id"]

            # If the full record fits, keep as-is; else split into multiple chunks to preserve all data
            candidate = {"id": base_id, "values": None, "metadata": base_meta}
            if _meta_size_bytes(base_meta) <= 40 * 1024:
                candidate["metadata"]["content_len"] = len(prefixed_text)
                final_records.append(candidate)
            else:
                chunks = split_record_into_chunks(
                    base_meta,
                    base_id,
                    prefixed_text,
                    limit_bytes=40 * 1024,
                    overlap_chars=256,
                    identity_header=identity_header,
                )
                for rec in chunks:
                    rec["metadata"]["content_len"] = len(rec["metadata"].get("content", ""))
                    final_records.append(rec)

        # Generate embeddings for all final texts
        final_texts = [r["metadata"]["content"] for r in final_records]
        vectors = embed_texts(oai, embedding_model, final_texts, batch_size=batch_size)

        if not vectors:
            logger.error("No vectors produced; check inputs")
            raise RuntimeError("No vectors produced; check inputs.")
        dim = len(vectors[0])
        logger.info(f"Generated {len(vectors)} embeddings with dimension {dim} for {len(final_records)} records")

        # Assign vectors to records
        for rec, vec in zip(final_records, vectors):
            rec["values"] = vec

        # Update records to final_records
        records = final_records

        # Note: keeping full chunk text in metadata (content) so Pinecone can return exact context directly.

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

    # upsert
    try:
        logger.info(f"Connecting to Pinecone index '{index_name}'")
        index = ensure_index(index_name, environment, dimension=dim)
        logger.info("Starting upsert process")
        total = 0
        for i in range(0, len(records), batch_size):
            part = records[i:i + batch_size]

            # Final safety: ensure metadata is still under 40KB (in case anything modified records later)
            for rec in part:
                ensure_metadata_limit(rec.get("metadata", {}), limit_bytes=40 * 1024, logger=logger)

            index.upsert(vectors=part, namespace=namespace)
            total += len(part)
            logger.info(f"Upserted batch {i // batch_size + 1}: {len(part)} records")
        logger.info(f"Upsert completed: {total} total records")
        print(f"[ok] Upserted {total} vectors to index='{index_name}', namespace='{namespace}'.")
    except Exception as e:
        logger.error(f"Failed to upsert records: {e}")
        raise

    # Finished
    logger.info("Ingestion process completed successfully")

if __name__ == "__main__":
    main()