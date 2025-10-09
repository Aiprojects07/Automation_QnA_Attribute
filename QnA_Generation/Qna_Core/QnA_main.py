#!/usr/bin/env python3
"""
Claude 4.1 (Claude Opus 4.1) QnA generator using **Anthropic's built-in Web Search tool**

What this script does
---------------------
- Reads a freeform **prompt file** (path passed as --prompt_path). You write the instructions there for how to generate Q&A.
- Reads a **CSV** with columns for **brand**, **product name**, and **shade of lipstick** (case-insensitive, flexible headers accepted).
- Calls Anthropic's Messages API with model **claude-opus-4-1-20250805** and the **web_search_20250305** tool enabled.
- For each input row, asks Claude to generate Q&A strictly following your JSON format and writes to **JSONL** (one JSON object per line).

Environment variables
---------------------
- `ANTHROPIC_API_KEY` (required)

Quick start
-----------
```bash
python claude_web_qna.py \
  --prompt_path prompt.md \
  --input_csv products.csv \
  --output_dir outputs
```

CSV headers
-----------
Accepted header names (case-insensitive):
- brand -> `brand`
- product name -> `product_name` (also accepts `product`, `product_title`, `line`, `product line`)
- shade of lipstick -> `shade_of_lipstick` (also accepts `shade`, `color`)

The model also receives a `full_name` we build like: `"{brand} {product_name} in {shade}"`.

Notes about the output JSON format
----------------------------------
If your JSON file looks like a template/example (e.g., keys such as `title`, `product.brand`, and an array of `sections` with `qas` objects), the script includes that **verbatim** in the message so Claude mirrors it exactly. If it looks like a formal JSON Schema, the script will still pass it through, but does not validate; the model is instructed to obey it strictly.

"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
import textwrap
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
import subprocess
import hashlib
import pandas as pd
from io import BytesIO
from openpyxl import load_workbook
from anthropic import Anthropic
from anthropic._exceptions import APIError as AnthropicAPIError
try:
    # Type helpers for batches (available in recent anthropic SDKs)
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as BatchRequest
except Exception:  # SDK may not expose types; we'll fall back to plain dicts
    MessageCreateParamsNonStreaming = None  # type: ignore
    BatchRequest = None  # type: ignore

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-4-1-20250805"  # From your example
# Default batch size for Anthropic Message Batches (change here if needed)
BATCH_SIZE_DEFAULT = 3
# Default temperature for both sync and batch calls
DEFAULT_TEMPERATURE = 1

# Control extra artifact creation without CLI args
WRITE_RAW_FULL: bool = False
WRITE_AUDIT_FILES: bool = False
USE_EXCEL_DATA: bool = False

# ------------------------------
# Logging Setup
# ------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration with both file and console handlers."""
    # Create logs directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"qna_generation_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# ------------------------------
# Helpers
# ------------------------------

def read_prompt(path: str) -> str:
    """Read prompt from file with fallback to sample prompt file."""
    logger = logging.getLogger(__name__)
    
    # Try to read the original prompt file
    if os.path.exists(path):
        logger.info(f"📄 Using prompt file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    else:
        # Fallback to sample prompt file
        sample_path = path.replace('lipstick-qa-prompt-builder.json', 'sample_prompt.json')
        if os.path.exists(sample_path):
            logger.warning(f"📄 Original prompt file not found. Using sample prompt file: {sample_path}")
            with open(sample_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        else:
            raise FileNotFoundError(f"Prompt file not found: {path} (and no sample fallback at {sample_path})")
    
    # Check if it's a JSON file with a "prompt" key
    if path.endswith('.json') or path.endswith('sample_prompt.json'):
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "prompt" in data:
                return data["prompt"]
        except json.JSONDecodeError:
            pass
    
    # Return raw content if not JSON or no "prompt" key
    return content


def convert_sharepoint_url(share_url: str) -> str:
    """Convert SharePoint share URL to direct download URL.
    Handles URLs like:
    https://<tenant>-my.sharepoint.com/:x:/g/personal/<user>/<SHARE_TOKEN>?e=...
    and produces:
    https://<tenant>-my.sharepoint.com/personal/<user>/_layouts/15/download.aspx?share=<SHARE_TOKEN>
    """
    try:
        # We only handle SharePoint patterns with /personal/ and an Office file marker like :x:
        if "/personal/" in share_url and ":" in share_url:
            from urllib.parse import urlsplit, quote as _urlquote
            sp = urlsplit(share_url)
            # Base domain like https://<tenant>-my.sharepoint.com
            base = f"{sp.scheme}://{sp.netloc}"
            # Split once at /personal/ to get user and token segment
            _, user_and_token = share_url.split("/personal/", 1)
            user_token_parts = user_and_token.split('/')
            if len(user_token_parts) >= 2:
                user_path = user_token_parts[0]
                share_token_raw = user_token_parts[1].split('?')[0]
                share_token = _urlquote(share_token_raw, safe='')
                download_url = f"{base}/personal/{user_path}/_layouts/15/download.aspx?share={share_token}"
                return download_url

        # OneDrive short links often require auth/redirect; return as-is
        if '1drv.ms' in share_url or 'onedrive.live.com' in share_url:
            return share_url
    except Exception:
        pass

    return share_url


def extract_first_json_object(text: str) -> str | None:
    """Heuristically extract the first top-level JSON object from a string.
    Returns the JSON substring if found, else None.
    """
    # Strip code fences if any remain
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    # Fast path: already valid JSON
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    # Find first '{' and attempt to match a balanced object
    start = s.find('{')
    if start == -1:
        return None

    # Use a simple brace counter to find the matching closing '}'
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = s[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    # continue scanning in case there's a later valid block
                    pass
    return None


def normalize_row(raw: Dict[str, str]) -> Dict[str, str]:
    # Normalize headers and values
    m = { (k or "").strip().lower().replace(" ", "_").replace(",", ""): (v or "").strip() for k, v in raw.items() }
    
    # Handle both Excel and CSV column mappings
    brand = m.get("brand") or m.get("company")
    
    # Product name mapping: Excel uses "lipstick_name", CSV uses "product_name"
    product_name = (
        m.get("product_name")
        or m.get("lipstick_name")  # Excel column
        or m.get("product")
        or m.get("product_title")
        or m.get("line")
        or m.get("product_line")
    )
    
    shade = m.get("shade_of_lipstick") or m.get("shade") or m.get("color")
    
    # Support multiple SKU header variants including Excel's 'Kult SKU Code'
    sku = (
        m.get("sku")
        or m.get("sku_code")
        or m.get("product_sku")
        or m.get("item_code")
        or m.get("kult_sku_code")  # Both Excel and CSV
        or ""
    )
    
    # Optional category support
    category = m.get("category") or m.get("product_category") or m.get("category_name") or ""
    
    # Sub-category mapping: Excel uses "sub_category", CSV uses "sub_category" 
    sub_category = (
        m.get("sub_category")
        or m.get("subcategory")
        or m.get("sub-category")
        or ""
    )
    
    # Leaf level category: Excel uses "sub_sub_category", CSV uses "sub_sub_category"
    leaf_level_category = (
        m.get("leaf_level_category")
        or m.get("leaflevelcategory")
        or m.get("leaf_level_cat")
        or m.get("sub_sub_category")  # Both Excel and CSV
        or ""
    )
    
    # CSV-specific color/appearance metrics (not in Excel)
    l_star = m.get("l*") or m.get("l_star") or m.get("l") or ""
    a_star = m.get("a*") or m.get("a_star") or m.get("a") or ""
    b_star = m.get("b*") or m.get("b_star") or m.get("b") or ""
    c_star = m.get("c*") or m.get("c_star") or m.get("c") or m.get("chroma") or ""
    h_deg = m.get("h°") or m.get("h_deg") or m.get("h") or ""
    sR = m.get("sr") or m.get("s_r") or ""
    sG = m.get("sg") or m.get("s_g") or ""
    sB = m.get("sb") or m.get("s_b") or ""
    gloss = m.get("gloss") or ""
    
    full_name = f"{brand} {product_name} {shade}".strip()
    
    s_no = m.get("s.no.") or m.get("s_no") or ""
    
    return {
        "brand": brand,
        "product_name": product_name,
        "shade_of_lipstick": shade,
        "sku": sku,
        "category": category,
        "sub_category": sub_category,
        "leaf_level_category": leaf_level_category,
        # CSV-specific color metrics
        "l_star": l_star,
        "a_star": a_star,
        "b_star": b_star,
        "c_star": c_star,
        "h_deg": h_deg,
        "sR": sR,
        "sG": sG,
        "sB": sB,
        "gloss": gloss,
        "full_name": full_name,
        "s_no": s_no,
    }


def read_rows(data_path: str) -> List[Dict[str, str]]:
    """Read product data from either Excel (SharePoint URL) or CSV file based on USE_EXCEL_DATA setting."""
    out: List[Dict[str, str]] = []
    
    if USE_EXCEL_DATA:
        # Handle Excel data from SharePoint URL
        try:
            # Convert SharePoint share URL to direct download URL
            logger = logging.getLogger(__name__)
            download_url = convert_sharepoint_url(data_path)
            try:
                logger.info("[excel] original_url=%s", data_path)
                logger.info("[excel] converted_download_url=%s", download_url)
            except Exception:
                pass
            
            # Download and read Excel file
            headers = {"User-Agent": "Mozilla/5.0 (compatible; QnA-Script/1.0)"}
            response = requests.get(download_url, headers=headers, timeout=30)
            if not response.ok:
                try:
                    err_snippet = response.text[:500] if response.text else None
                    logger.error("[excel] GET failed: status=%s url=%s body_snippet=%r", response.status_code, response.url, err_snippet)
                except Exception:
                    pass
                # Fallback: some tenants use guestaccess.aspx instead of download.aspx
                if response.status_code == 404 and 'download.aspx' in download_url:
                    alt_url = download_url.replace('download.aspx', 'guestaccess.aspx')
                    try:
                        logger.info("[excel] retrying with guestaccess.aspx: %s", alt_url)
                    except Exception:
                        pass
                    response = requests.get(alt_url, headers=headers, timeout=30)
                # If still not ok, raise
                if not response.ok:
                    response.raise_for_status()
            
            # Read Excel file from memory
            excel_data = BytesIO(response.content)
            df = pd.read_excel(excel_data, engine='openpyxl')
            
            # Convert DataFrame to list of dictionaries
            for i, (_, row) in enumerate(df.iterrows(), 1):
                try:
                    # Convert pandas Series to dictionary, handling NaN values
                    row_dict = {k: (str(v) if pd.notna(v) else "") for k, v in row.items()}
                    out.append(normalize_row(row_dict))
                except Exception as e:
                    raise ValueError(f"Excel row {i} invalid: {e}") from e
                    
        except Exception as e:
            raise ValueError(f"Failed to read Excel data from {data_path}: {e}") from e
    else:
        # Handle CSV data from local file
        with open(data_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader, 1):
                try:
                    out.append(normalize_row(r))
                except Exception as e:
                    raise ValueError(f"CSV row {i} invalid: {e}") from e
    
    if not out:
        raise ValueError(f"Input data contained no rows: {data_path}")
    return out


def sanitize_filename(text: str) -> str:
    """Convert text to a safe filename by removing/replacing problematic characters."""
    import re
    # Replace spaces and problematic characters with hyphens
    text = re.sub(r'[^\w\s-]', '', text)  # Remove special chars except word chars, spaces, hyphens
    text = re.sub(r'[-\s]+', '-', text)   # Replace multiple spaces/hyphens with single hyphen
    return text.strip('-').lower()        # Remove leading/trailing hyphens and convert to lowercase


def create_output_filename(product_row: Dict[str, str]) -> str:
    """Create a filename in format: sku-brand-product-name-shade.json"""
    sku = sanitize_filename(product_row.get("sku", "")) if product_row.get("sku") else ""
    brand = sanitize_filename(product_row["brand"])
    product_name = sanitize_filename(product_row["product_name"])
    shade = sanitize_filename(product_row["shade_of_lipstick"])
    
    if sku:
        return f"{sku}-{brand}-{product_name}-{shade}.json"
    else:
        return f"{brand}-{product_name}-{shade}.json"


def trigger_ingestion(json_path: str, logger: logging.Logger) -> bool:
    """Run the Pinecone ingestion script for the given JSON path.
    Returns True on success, False on failure.
    """
    try:
        env = os.environ.copy()
        env["FILE_PATH"] = json_path
        script_path = "/home/sid/Documents/Automation_QnA_Attribute/clustering_Pinecone/pinecone_core/clustering_upserting.py"
        logger.info(f"Triggering ingestion for: {json_path}")
        result = subprocess.run([sys.executable, script_path], env=env, capture_output=True, text=True)
        if result.returncode != 0:
            # Log full stderr/stdout for troubleshooting
            if result.stdout:
                for line in result.stdout.splitlines():
                    logger.info("[ingest][stdout] %s", line)
            if result.stderr:
                for line in result.stderr.splitlines():
                    logger.error("[ingest][stderr] %s", line)
            logger.error("Ingestion failed for %s: returncode=%s", json_path, result.returncode)
            return False
        # Log full stdout for traceability
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info("[ingest][stdout] %s", line)
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning("[ingest][stderr] %s", line)
        logger.info("Ingestion completed for %s", json_path)
        return True
    except Exception as e:
        logger.error("Exception while triggering ingestion for %s: %s", json_path, e)
        return False


# ------------------------------
# Batch mode (Anthropic Message Batches API)
# ------------------------------

def _build_shared_system_blocks(prompt_text: str, use_cache: bool) -> List[Dict[str, Any]]:
    """Return a system array with a small uncached header + a large cached block.
    We keep the large block byte-stable and mark it with cache_control so batches can reuse it.
    Always uses the inline REQUIRED JSON FORMAT (no external schema).
    """
    # Small uncached header (stable too, but usually tiny)
    header = {
        "type": "text",
        "text": textwrap.dedent(
            """
            You are an expert Q&A generator for cosmetics. Follow the formatting and constraints exactly.
            Output strictly valid JSON only. No markdown.
            """
        ).strip() + "\n"
    }

    # Large cached template: prompt + required JSON SHAPE (inline only)
    required_format = textwrap.dedent(
        """
        {
          "product": {
            "brand": "...",
            "product_line": "...",
            "shade": "...",
            "full_name": "...",
            "sku": "...",
            "category": "...",
            "sub_category": "...",
            "leaf_level_category": "..."
          },
          "sections": [
            {
              "title": "Section title here",
              "qas": [
                {
                  "q": "question here",
                  "a": "answer here",
                  "why": "explanation here",
                  "solution": "solution here (include this key even if empty)",
                  "CONFIDENCE": "High/Medium/Low | Source: [Review consensus from X+ reviews/Ingredient analysis/Limited data] | Context: [specific reasoning]"
                }
              ]
            }
          ]
        }
        """
    ).strip()

    big_block = {
        "type": "text",
        "text": textwrap.dedent(
            f"""
            {prompt_text}

            CRITICAL FORMATTING REQUIREMENTS:
            - Return ONLY valid JSON (no markdown, no commentary)
            - Treat the JSON below as a SHAPE EXAMPLE (keys and nesting) only; do NOT copy the number of sections or QAs from it
            - Do NOT include code fences (```), markdown, prose, or any text before/after the JSON
            - Follow the section and QA counts from the PROMPT'S STRUCTURE 
            - Expand arrays to meet the PROMPT requirements even if the example shows fewer items
            - Use the exact product keys shown (including "sku", "category", "sub_category", "leaf_level_category"); keys must always be present (empty string allowed if unknown)
            - Do NOT include any citations, footnotes, source markers, or attribution (e.g., <cite ...>...</cite>, [1], (ref), URLs). Present everything as expert knowledge.

            REQUIRED JSON FORMAT:
            {required_format}
            """
        ).strip(),
    }
    # Attach cache control only when caching is enabled
    # DIAGNOSTIC: log what use_cache actually is here
    try:
        logging.getLogger(__name__).info("_build_shared_system_blocks: use_cache=%r (type=%s)", use_cache, type(use_cache).__name__)
    except Exception:
        pass
    if use_cache:
        big_block["cache_control"] = {"type": "ephemeral"}
    return [header, big_block]


def _make_custom_id(row: Dict[str, str]) -> str:
    sku = (row.get("sku") or "").strip()
    if sku:
        return sku
    return f"{(row.get('brand') or '').strip()}-{(row.get('product_name') or '').strip()}-{(row.get('shade_of_lipstick') or '').strip()}" or "unknown"


def run_batch_generation(
    api_key: str,
    model: str,
    rows: List[Dict[str, str]],
    prompt_text: str,
    max_tokens: int,
    batch_size: int,
    output_dir: str,
    logger: logging.Logger,
    no_ingest: bool = False,
    use_cache: bool = True,
) -> Tuple[int, int, int, int]:
    """Send rows to Anthropic Message Batches in groups and process results.
    Returns (ok_count, fail_count, total_input_tokens, total_output_tokens).
    """
    # Set beta header on the client (batches reject a 'betas' body param). This enables web_search tool in batches.
    client = Anthropic(api_key=api_key, default_headers={"anthropic-beta": "web-search-2025-03-05"})
    ok, fail = 0, 0
    total_input_tokens, total_output_tokens = 0, 0

    shared_system = _build_shared_system_blocks(prompt_text, use_cache)
    try:
        logger.info("Prompt caching (batch): %s", "ENABLED" if use_cache else "DISABLED")
    except Exception:
        pass

    for i in range(0, len(rows), batch_size):
        wave = rows[i:i+batch_size]
        requests_payload: List[Any] = []
        
        # Debug: Log the wave contents to identify duplicates
        logger.info(f"Processing wave {i//batch_size + 1}: {len(wave)} items")
        for idx, row in enumerate(wave):
            custom_id = _make_custom_id(row)
            logger.info(f"  Wave item {idx}: custom_id={custom_id}, brand={row.get('brand')}, product={row.get('product_name')}, shade={row.get('shade_of_lipstick')}")
        
        # Check for duplicate custom_ids within this wave
        custom_ids_in_wave = [_make_custom_id(row) for row in wave]
        if len(custom_ids_in_wave) != len(set(custom_ids_in_wave)):
            logger.error(f"DUPLICATE custom_ids detected in wave: {custom_ids_in_wave}")
            duplicates = [x for x in custom_ids_in_wave if custom_ids_in_wave.count(x) > 1]
            logger.error(f"Duplicate IDs: {set(duplicates)}")
        
        for row in wave:
            # Build per-product user message (JSON product info)
            _, user_text = build_user_message(prompt_text, row, use_natural_generation=True)

            # If typed classes are available, use them; else fall back to dicts
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": DEFAULT_TEMPERATURE,
                "system": shared_system,
                "messages": [{"role": "user", "content": user_text}],
                # Ensure batch requests mirror the sync path by enabling web_search tool
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 15,
                    }
                ],
                # Enable structured "thinking" with a high token budget per user specs
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 31999,
                },
            }
            custom_id = _make_custom_id(row)

            if MessageCreateParamsNonStreaming and BatchRequest:
                requests_payload.append(
                    BatchRequest(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(**params)
                    )
                )
            else:
                # Fallback compatible with SDK expecting dicts
                requests_payload.append({"custom_id": custom_id, "params": params})

        try:
            logger.info("Submitting batch: %s requests (items %s-%s)", len(requests_payload), i+1, i+len(wave))
            batch = client.messages.batches.create(requests=requests_payload)
            batch_id = getattr(batch, "id", None) or batch["id"]  # type: ignore
            logger.info("Batch submitted: id=%s", batch_id)

            # Poll until completed
            while True:
                status_obj = client.messages.batches.retrieve(batch_id)
                status = getattr(status_obj, "processing_status", None)
                logger.info("Batch %s status: %s", batch_id, status)
                if status is None:
                    logger.warning("Batch %s status response missing processing_status: %r", batch_id, status_obj)
                    break
                if status in ("ended", "completed", "cancelled", "expired", "failed"):
                    break
                time.sleep(3)

            # Fetch results once: materialize JSONL decoder and log per-item errors
            results_iter = client.messages.batches.results(batch_id)
            try:
                raw_results = list(results_iter)  # JSONL decoder -> list of items
            except Exception:
                # Fallback for SDKs that return an object with .data
                raw_results = getattr(results_iter, "data", None) or results_iter.get("data", [])  # type: ignore

            # First pass over concrete list: surface errors
            for item in raw_results:
                try:
                    custom_id_iter = getattr(item, "custom_id", None)
                    result_obj = getattr(item, "result", None)
                    # SDK shape: errored -> result.error.error.message
                    err_wrapper = getattr(result_obj, "error", None)
                    err_inner = getattr(err_wrapper, "error", None)
                    err_msg = (
                        getattr(err_inner, "message", None)
                        or getattr(err_wrapper, "message", None)
                        or None
                    )
                    if err_msg:
                        logger.error("Batch item failed: %s -> %s", custom_id_iter, err_msg)
                except Exception:
                    # Best-effort logging; don't break on a single malformed item
                    continue

            items = raw_results
            logger.info("Batch %s returned %s results", batch_id, len(items))
            # Prepare a queue for items that require resume (to run as a SECOND BATCH)
            resume_queue: List[Dict[str, Any]] = []

            # Process each result item
            for item in items:
                custom_id = getattr(item, "custom_id", None)
                result_obj = getattr(item, "result", None)

                # If not succeeded, mark as failed and continue
                if getattr(result_obj, "type", None) != "succeeded":
                    # Try to log an error message
                    err_wrapper = getattr(result_obj, "error", None)
                    err_inner = getattr(err_wrapper, "error", None)
                    err_msg = (
                        getattr(err_inner, "message", None)
                        or getattr(err_wrapper, "message", None)
                        or None
                    )
                    if err_msg:
                        logger.error("Batch item failed for %s: %s", custom_id, err_msg)
                    # Update checkpoint for this row
                    row = next((r for r in wave if _make_custom_id(r) == custom_id), None)
                    if row:
                        update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
                    fail += 1
                    continue

                # Succeeded: get the message payload
                message = getattr(result_obj, "message", None)

                # Try to find the corresponding row
                row = next((r for r in wave if _make_custom_id(r) == custom_id), None)
                if not row:
                    logger.warning("Result with custom_id=%s has no matching input row", custom_id)
                    fail += 1
                    continue

                filename = create_output_filename(row)
                filepath = os.path.join(output_dir, filename)

                # Write full raw batch item as plain text (serialized JSON) sidecar
                try:
                    base_no_ext = os.path.splitext(filename)[0]
                    raw_result_txt_path = os.path.join(output_dir, f"{base_no_ext}.raw.result.txt")
                    raw_serialized = _to_json_serializable(item)
                    with open(raw_result_txt_path, "w", encoding="utf-8") as f_rawtxt:
                        f_rawtxt.write(json.dumps(raw_serialized, ensure_ascii=False, indent=2))
                except Exception as _:
                    logger.debug("Failed to write raw result text sidecar for %s", filename)

                # Cache metrics per item (best-effort: try multiple locations)
                try:
                    cache_status = None
                    cache_token_credits = None
                    usage = getattr(message, "usage", None)
                    # Some SDKs surface headers or metadata differently; probe common spots
                    headers = (
                        getattr(message, "response_headers", None)
                        or getattr(message, "headers", None)
                        or getattr(item, "response_headers", None)
                        or getattr(item, "headers", None)
                        or {}
                    )
                    if isinstance(headers, dict):
                        cache_status = headers.get("anthropic-cache-status") or headers.get("x-anthropic-cache-status")
                        cache_token_credits = headers.get("anthropic-cache-token-credits")
                    # In some cases, cache info may be copied into usage
                    if isinstance(usage, dict):
                        cache_status = cache_status or usage.get("anthropic-cache-status")
                        cache_token_credits = cache_token_credits or usage.get("anthropic-cache-token-credits")
                    if cache_status or cache_token_credits:
                        logger.info(
                            "[batch][cache] custom_id=%s status=%s token_credits=%s", custom_id, cache_status, cache_token_credits
                        )
                    else:
                        logger.info("[batch][cache] custom_id=%s status=%s", custom_id, None)
                except Exception as _:
                    logger.debug("Failed to read cache headers for custom_id=%s", custom_id)

                # Extract text parts from non-streaming result
                try:
                    content_blocks = getattr(message, "content", None) or []
                    parts = []
                    for c in content_blocks:
                        if getattr(c, "type", None) == "text":
                            parts.append(getattr(c, "text", ""))
                    # Fallback: if no explicit 'text' blocks, try SDK convenience fields and log block types
                    if not parts:
                        try:
                            content_types = [getattr(c, "type", type(c).__name__) for c in content_blocks]
                            logger.warning("[batch] No 'text' blocks found. content_types=%s", content_types)
                        except Exception:
                            pass
                        # Some SDK versions expose a unified text via 'output_text' or 'text'
                        try:
                            fallback_text = getattr(message, "output_text", None) or getattr(message, "text", None)
                            if isinstance(fallback_text, str) and fallback_text.strip():
                                parts = [fallback_text]
                                logger.info("[batch] Used message.output_text/text as fallback for custom_id=%s", custom_id)
                        except Exception:
                            pass

                    # Resume policy:
                    # - ALWAYS resume when stop_reason == "pause_turn" (even if partial text exists)
                    # - ALSO resume when there are no text parts (defensive for other anomalies)
                    try:
                        stop_reason = getattr(message, "stop_reason", None)
                    except Exception:
                        stop_reason = None
                    if stop_reason == "pause_turn":
                        # Queue for parallel resume via a second batch
                        try:
                            _, user_text_resume = build_user_message(prompt_text, row, use_natural_generation=True)
                        except Exception:
                            user_text_resume = ""
                        resume_queue.append({
                            "custom_id": custom_id,
                            "row": row,
                            "original_user_text": user_text_resume,
                            "combined_blocks": _clean_orphan_tool_use(_to_json_serializable(content_blocks)),
                        })
                        # Skip finalization for now; will be handled in resume batch
                        continue
                    elif not parts:
                        # Queue for parallel resume via a second batch
                        try:
                            _, user_text_resume = build_user_message(prompt_text, row, use_natural_generation=True)
                        except Exception:
                            user_text_resume = ""
                        resume_queue.append({
                            "custom_id": custom_id,
                            "row": row,
                            "original_user_text": user_text_resume,
                            "combined_blocks": _clean_orphan_tool_use(_to_json_serializable(content_blocks)),
                        })
                        continue
                    
                    # If enabled, write raw text response (batch mode)
                    if WRITE_RAW_FULL:
                        try:
                            base_no_ext = os.path.splitext(filename)[0]
                            raw_sidecar_path = os.path.join(output_dir, f"{base_no_ext}.raw.txt")
                            
                            # Extract all text content from the response
                            all_text_parts = []
                            for c in content_blocks:
                                if getattr(c, "type", None) == "text":
                                    all_text_parts.append(getattr(c, "text", ""))
                            # Also include fallback_text if no text blocks were present
                            if not all_text_parts:
                                try:
                                    fb_txt = getattr(message, "output_text", None) or getattr(message, "text", None)
                                    if isinstance(fb_txt, str) and fb_txt.strip():
                                        all_text_parts.append(fb_txt)
                                except Exception:
                                    pass
                            
                            # Save complete raw text response
                            raw_text = "\n".join(all_text_parts)
                            with open(raw_sidecar_path, "w", encoding="utf-8") as f_side:
                                f_side.write(raw_text)
                        except Exception as _:
                            logger.debug("Failed to write raw text dump for %s", filename)

                    # Use only the FINAL text block as the model's answer
                    text = (parts[-1] if parts else "").strip()
                    raw_for_parse = extract_first_json_object(text) or text
                    obj = json.loads(raw_for_parse)
                    # Accumulate token usage if present
                    usage = getattr(message, "usage", None)
                    try:
                        if isinstance(usage, dict):
                            in_tok = int(usage.get("input_tokens", 0) or 0)
                            out_tok = int(usage.get("output_tokens", 0) or 0)
                        else:
                            # Anthropic SDK Usage is a Pydantic model; read attributes
                            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
                            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
                        total_input_tokens += in_tok
                        total_output_tokens += out_tok
                        try:
                            logger.info("[batch][usage] custom_id=%s in=%s out=%s", custom_id, in_tok, out_tok)
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception as e:
                    logger.error("Failed to parse batch item %s JSON: %s", custom_id, e)
                    update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
                    fail += 1
                    continue

                # Save output JSON
                try:
                    with open(filepath, "w", encoding="utf-8") as out:
                        json.dump(obj, out, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error("Failed to write output for %s: %s", filename, e)
                    update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
                    fail += 1
                    continue

                # Trigger ingestion (honor no_ingest like the sync path)
                ing_ok = True
                if not no_ingest:
                    ing_ok = trigger_ingestion(filepath, logger)
                else:
                    logger.info("Auto-ingest disabled via --no_ingest; skipping ingestion for %s", filepath)
                update_checkpoint(output_dir, row, ing_ok, load_checkpoint(output_dir))
                if ing_ok:
                    ok += 1
                else:
                    fail += 1

            # Process resume_queue in parallel via batch waves until all are terminal
            while resume_queue:
                try:
                    logger.info("Starting resume batch for %s items", len(resume_queue))
                except Exception:
                    pass
                resume_requests: List[Any] = []
                # Build resume requests
                for r in resume_queue:
                    custom_id_r = r["custom_id"]
                    row_r = r["row"]
                    original_user_text_r = r.get("original_user_text", "")
                    combined_blocks_r = r.get("combined_blocks", [])
                    # Extra safety: sanitize combined blocks before sending
                    combined_blocks_r = _clean_orphan_tool_use(combined_blocks_r)
                    messages_r = [
                        {"role": "user", "content": original_user_text_r},
                        {"role": "assistant", "content": combined_blocks_r},
                        {"role": "user", "content": "Please continue and return ONLY the final JSON now. Use the existing search results above"},
                    ]
                    params_r = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": DEFAULT_TEMPERATURE,
                        "system": shared_system,
                        "messages": messages_r,
                        "tools": [
                            {
                                "type": "web_search_20250305",
                                "name": "web_search",
                                "max_uses": 15,
                            }
                        ],
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 31999,
                        },
                    }
                    if MessageCreateParamsNonStreaming and BatchRequest:
                        resume_requests.append(
                            BatchRequest(
                                custom_id=custom_id_r,
                                params=MessageCreateParamsNonStreaming(**params_r)
                            )
                        )
                    else:
                        resume_requests.append({"custom_id": custom_id_r, "params": params_r})

                # Submit resume batch
                try:
                    resume_batch = client.messages.batches.create(requests=resume_requests)
                    resume_batch_id = getattr(resume_batch, "id", None) or resume_batch["id"]  # type: ignore
                    logger.info("Resume batch submitted: id=%s", resume_batch_id)
                    # Poll
                    while True:
                        st = client.messages.batches.retrieve(resume_batch_id)
                        st_status = getattr(st, "processing_status", None)
                        logger.info("Resume batch %s status: %s", resume_batch_id, st_status)
                        if st_status is None or st_status in ("ended", "completed", "cancelled", "expired", "failed"):
                            break
                        time.sleep(3)
                    # Fetch results
                    resume_results_iter = client.messages.batches.results(resume_batch_id)
                    try:
                        resume_items = list(resume_results_iter)
                    except Exception:
                        resume_items = getattr(resume_results_iter, "data", None) or resume_results_iter.get("data", [])  # type: ignore
                except Exception as e_res:
                    logger.error("Resume batch submission failed: %s", e_res)
                    # Mark all queued items as failed in checkpoint and count as fail
                    for r in resume_queue:
                        update_checkpoint(output_dir, r["row"], False, load_checkpoint(output_dir))
                        fail += 1
                    resume_queue = []
                    break

                # Map custom_id to queue entry for quick access
                resume_map = {r["custom_id"]: r for r in resume_queue}
                next_round: List[Dict[str, Any]] = []

                # Process each resume item
                for item_r in resume_items:
                    custom_id_r = getattr(item_r, "custom_id", None)
                    result_obj_r = getattr(item_r, "result", None)
                    if getattr(result_obj_r, "type", None) != "succeeded":
                        # mark fail
                        r = resume_map.get(custom_id_r)
                        if r:
                            update_checkpoint(output_dir, r["row"], False, load_checkpoint(output_dir))
                            fail += 1
                        continue
                    message_r = getattr(result_obj_r, "message", None)
                    content_blocks_r = getattr(message_r, "content", None) or []
                    stop_reason_r = getattr(message_r, "stop_reason", None)
                    # Extend prior combined blocks
                    prior = resume_map.get(custom_id_r)
                    if not prior:
                        continue
                    combined_so_far = prior.get("combined_blocks", [])
                    combined_so_far.extend(_clean_orphan_tool_use(content_blocks_r))

                    # Usage accumulation
                    usage_r = getattr(message_r, "usage", None)
                    try:
                        if isinstance(usage_r, dict):
                            in_tok = int(usage_r.get("input_tokens", 0) or 0)
                            out_tok = int(usage_r.get("output_tokens", 0) or 0)
                        else:
                            in_tok = int(getattr(usage_r, "input_tokens", 0) or 0)
                            out_tok = int(getattr(usage_r, "output_tokens", 0) or 0)
                        total_input_tokens += in_tok
                        total_output_tokens += out_tok
                        logger.info("[batch][resume][usage] custom_id=%s in+=%s out+=%s", custom_id_r, in_tok, out_tok)
                    except Exception:
                        pass

                    if stop_reason_r == "pause_turn":
                        # Queue again with extended blocks
                        next_round.append({
                            "custom_id": custom_id_r,
                            "row": prior["row"],
                            "original_user_text": prior.get("original_user_text", ""),
                            "combined_blocks": combined_so_far,
                        })
                        continue

                    # Terminal: finalize -> extract text, parse, write, ingest, checkpoint
                    r_row = prior["row"]
                    filename = create_output_filename(r_row)
                    filepath = os.path.join(output_dir, filename)

                    # Extract and merge all text blocks
                    resumed_parts = [b.get("text", "") for b in combined_so_far if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
                    text_final = ("\n".join(resumed_parts)).strip() if resumed_parts else ""
                    raw_for_parse = extract_first_json_object(text_final) or text_final
                    try:
                        obj = json.loads(raw_for_parse)
                    except Exception as e_json:
                        logger.error("Failed to parse resumed JSON for %s: %s", custom_id_r, e_json)
                        update_checkpoint(output_dir, r_row, False, load_checkpoint(output_dir))
                        fail += 1
                        continue

                    # Optional raw text dump
                    if WRITE_RAW_FULL:
                        try:
                            base_no_ext = os.path.splitext(filename)[0]
                            raw_sidecar_path = os.path.join(output_dir, f"{base_no_ext}.raw.txt")
                            with open(raw_sidecar_path, "w", encoding="utf-8") as f_side:
                                f_side.write(text_final)
                        except Exception:
                            logger.debug("Failed to write resume raw text for %s", filename)

                    # Write output JSON
                    try:
                        with open(filepath, "w", encoding="utf-8") as out:
                            json.dump(obj, out, ensure_ascii=False, indent=2)
                    except Exception as e_write:
                        logger.error("Failed to write output for %s: %s", filename, e_write)
                        update_checkpoint(output_dir, r_row, False, load_checkpoint(output_dir))
                        fail += 1
                        continue

                    # Ingest
                    ing_ok = True
                    if not no_ingest:
                        ing_ok = trigger_ingestion(filepath, logger)
                    else:
                        logger.info("Auto-ingest disabled via --no_ingest; skipping ingestion for %s", filepath)
                    update_checkpoint(output_dir, r_row, ing_ok, load_checkpoint(output_dir))
                    if ing_ok:
                        ok += 1
                    else:
                        fail += 1

                # Prepare for next round if any still paused
                resume_queue = next_round

        except AnthropicAPIError as e:
            logger.error("Batch API error: %s", e)
            # Mark all rows in this wave as failed in the checkpoint so reruns track them
            for row in wave:
                update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
            fail += len(wave)
        except Exception as e:
            logger.error("Unexpected error during batch submission/processing: %s", e)
            for row in wave:
                update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
            fail += len(wave)

    return ok, fail, total_input_tokens, total_output_tokens


# --- Utility: safely convert SDK/Pydantic objects to JSON-serializable data ---
def _to_json_serializable(obj):
    """Best-effort conversion of SDK objects to JSON-serializable structures.
    Keeps dicts/lists/primitives as-is, tries model_dump()/dict()/vars(), then str().
    """
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # List/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    # Dict
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    # Pydantic v2
    try:
        return _to_json_serializable(obj.model_dump())  # type: ignore[attr-defined]
    except Exception:
        pass
    # Pydantic v1
    try:
        return _to_json_serializable(obj.dict())  # type: ignore[attr-defined]
    except Exception:
        pass
    # Generic objects
    try:
        return {k: _to_json_serializable(v) for k, v in vars(obj).items()}
    except Exception:
        pass
    # Fallback to string
    try:
        return str(obj)
    except Exception:
        return None


# ------------------------------
# Checkpoint System
# ------------------------------

def load_checkpoint(output_dir: str) -> Dict[str, Any]:
    """Load checkpoint data from file."""
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure expected keys exist (backward-compatible with older checkpoints)
                data.setdefault("completed_products", [])
                data.setdefault("completed_skus", [])
                data.setdefault("failed_products", [])
                data.setdefault("failed_skus", [])
                data.setdefault("total_processed", 0)
                data.setdefault("successful", 0)
                data.setdefault("failed", 0)
                data.setdefault("last_updated", None)
                data.setdefault("session_stats", [])

                # Convert list-to-set where appropriate
                if isinstance(data.get("completed_skus"), list):
                    data["completed_skus"] = set(data["completed_skus"])
                if isinstance(data.get("failed_skus"), list):
                    data["failed_skus"] = set(data["failed_skus"])
                return data
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load checkpoint: {e}")
    
    return {
        "completed_products": [],
        "completed_skus": set(),
        "failed_products": [],
        "failed_skus": set(),
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "last_updated": None,
        "session_stats": []
    }


def save_checkpoint(output_dir: str, checkpoint_data: Dict[str, Any]) -> None:
    """Save checkpoint data to file."""
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    # Convert set to list for JSON serialization
    checkpoint_copy = checkpoint_data.copy()
    if "completed_skus" in checkpoint_copy:
        checkpoint_copy["completed_skus"] = list(checkpoint_copy["completed_skus"])
    if "failed_skus" in checkpoint_copy:
        checkpoint_copy["failed_skus"] = list(checkpoint_copy["failed_skus"])
    
    checkpoint_copy["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_copy, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save checkpoint: {e}")


def update_checkpoint(output_dir: str, product_row: Dict[str, str], success: bool, checkpoint_data: Dict[str, Any]) -> None:
    """Update checkpoint with completed product."""
    product_id = f"{product_row['brand']}_{product_row['product_name']}_{product_row['shade_of_lipstick']}"
    if product_row.get('sku'):
        product_id += f"_{product_row['sku']}"
    
    # Maintain completed vs failed lists/sets
    if success:
        if product_id not in checkpoint_data.get("completed_products", []):
            checkpoint_data["completed_products"].append(product_id)
        # remove from failed if previously present
        if product_id in checkpoint_data.get("failed_products", []):
            checkpoint_data["failed_products"].remove(product_id)
        if product_row.get('sku'):
            checkpoint_data.setdefault("completed_skus", set()).add(product_row['sku'])
            checkpoint_data.setdefault("failed_skus", set()).discard(product_row['sku'])
    else:
        # record failure for visibility and future reprocessing
        if product_id not in checkpoint_data.get("failed_products", []):
            checkpoint_data.setdefault("failed_products", []).append(product_id)
        # do not add to completed on failure; ensure it's not marked completed spuriously
        if product_id in checkpoint_data.get("completed_products", []):
            checkpoint_data["completed_products"].remove(product_id)
        if product_row.get('sku'):
            checkpoint_data.setdefault("failed_skus", set()).add(product_row['sku'])
            checkpoint_data.setdefault("completed_skus", set()).discard(product_row['sku'])
    
    checkpoint_data["total_processed"] += 1
    if success:
        checkpoint_data["successful"] += 1
    else:
        checkpoint_data["failed"] += 1
    
    save_checkpoint(output_dir, checkpoint_data)


def is_product_completed(product_row: Dict[str, str], checkpoint_data: Dict[str, Any]) -> bool:
    """Check if product is already completed."""
    # Check by SKU first (most reliable)
    if product_row.get('sku') and product_row['sku'] in checkpoint_data.get("completed_skus", set()):
        return True
    
    # Check by product identifier
    product_id = f"{product_row['brand']}_{product_row['product_name']}_{product_row['shade_of_lipstick']}"
    if product_row.get('sku'):
        product_id += f"_{product_row['sku']}"
    
    return product_id in checkpoint_data.get("completed_products", [])


def build_user_message(prompt_text: str, product_row: Dict[str, str], use_natural_generation: bool = True) -> Tuple[str, str]:
    # Map CSV fields to the schema keys commonly used in your lipstick template
    product_for_model = {
        "brand": product_row["brand"],
        "product_line": product_row["product_name"],  # maps to template's product_line
        "shade": product_row["shade_of_lipstick"],
        "full_name": product_row["full_name"],
        "sku": product_row.get("sku", ""),
        "category": product_row.get("category", ""),
        "sub_category": product_row.get("sub_category", ""),
        "leaf_level_category": product_row.get("leaf_level_category", ""),
        # Pass-through optional dynamic metrics (may be empty strings)
        "l_star": product_row.get("l_star", ""),
        "a_star": product_row.get("a_star", ""),
        "b_star": product_row.get("b_star", ""),
        "c_star": product_row.get("c_star", ""),
        "h_deg": product_row.get("h_deg", ""),
        "sR": product_row.get("sR", ""),
        "sG": product_row.get("sG", ""),
        "sB": product_row.get("sB", ""),
        "gloss": product_row.get("gloss", ""),
        "s_no": product_row.get("s_no", ""),
    }

    if use_natural_generation:
        # Build a STATIC system message (cacheable) with prompt_text only
        # The OUTPUT_FORMAT is already included in the prompt file
        system_text = textwrap.dedent(
            f"""
            {prompt_text}

            CRITICAL FORMATTING REQUIREMENTS:
            - Return ONLY valid JSON (no markdown, no commentary)
            - Follow the OUTPUT_FORMAT structure defined in the prompt
            - Do NOT include code fences (```), markdown, prose, or any text before/after the JSON
            - Follow the section and QA counts from the PROMPT'S STRUCTURE 
            - Expand arrays to meet the PROMPT requirements even if the example shows fewer items
            - Use the exact product keys shown (including "sku", "category", "sub_category", "leaf_level_category"); keys must always be present (empty string allowed if unknown)
            - Do NOT include any citations, footnotes, source markers, or attribution (e.g., <cite ...>...</cite>, [1], (ref), URLs). Present everything as expert knowledge.
            """
        ).strip()

        # Dynamic per-product message (non-cacheable)
        user_text = textwrap.dedent(
            f"""
            Product Information:
            {json.dumps(product_for_model, ensure_ascii=False, indent=2)}
            """
        ).strip()

        return system_text, user_text
    else:
        raise ValueError("use_natural_generation=False is not supported. Refusing to build an alternate prompt shape.")



# ------------------------------
# Anthropic call w/ built-in web search tool
# ------------------------------

def call_claude_with_web_search(api_key: str, model: str, user_content: str, max_tokens: int = 32000, debug_basepath: str | None = None, system_text: str | None = None, use_cache: bool = True) -> tuple[str, dict]:
    """Call Claude API with retry logic and usage tracking."""
    logger = logging.getLogger(__name__)
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # Add prompt caching beta header only when caching is enabled
    if use_cache:
        headers["anthropic-beta"] = "prompt-caching-2024-07-31"
    # Headers prepared
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": DEFAULT_TEMPERATURE,
        "messages": [{"role": "user", "content": user_content}],
        # IMPORTANT: Use Anthropic's built-in web search tool per your example
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }
        ],
    }
    
    # Inject cacheable system message (prompt caching) if provided
    if system_text:
        # Log a stable hash of the system text to ensure byte-for-byte stability across requests
        try:
            sys_hash = hashlib.sha256((system_text or "").encode("utf-8")).hexdigest()
            logger.info("System text hash: %s", sys_hash)
        except Exception:
            pass

        block = {
            "type": "text",
            "text": system_text,
        }
        if use_cache:
            block["cache_control"] = {"type": "ephemeral"}
        payload["system"] = [block]
    
    # Log a concise request summary (avoid logging full prompt to save tokens in logs)
    try:
        logger.info(
            "LLM Request -> model=%s, max_tokens=%s, temperature=%.2f, tools=[%s], msg_chars=%s",
            payload["model"], payload["max_tokens"], payload.get("temperature", 0),
            ",".join(t.get("name", "?") for t in payload.get("tools", [])),
            len(user_content or "")
        )
    except Exception:
        pass

    # Optional debug: dump request payload and user message
    if debug_basepath:
        try:
            os.makedirs(os.path.dirname(debug_basepath), exist_ok=True)
            with open(f"{debug_basepath}_request_payload.json", "w", encoding="utf-8") as f_req:
                json.dump(payload, f_req, ensure_ascii=False, indent=2)
            with open(f"{debug_basepath}_user_message.txt", "w", encoding="utf-8") as f_msg:
                f_msg.write(user_content)
        except Exception as _:
            logger.debug("Failed to write debug request files for %s", debug_basepath)

    # Single-attempt call (no retry logic)
    try:
        start_time = time.time()
        # NOTE: Timeout removed per user request; be aware this may hang indefinitely on network stalls.
        url = ANTHROPIC_API_URL
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Read cache-related response headers for visibility
        cache_status = resp.headers.get("anthropic-cache-status")
        cache_token_credits = resp.headers.get("anthropic-cache-token-credits")

        # Optional debug: dump raw response JSON
        if debug_basepath:
            try:
                with open(f"{debug_basepath}_response_raw.json", "w", encoding="utf-8") as f_raw:
                    json.dump(data, f_raw, ensure_ascii=False, indent=2)
            except Exception:
                logger.debug("Failed to write raw response for %s", debug_basepath)
        
        # Extract usage information
        usage_info = data.get("usage", {})
        # Surface cache headers into usage_info for downstream logging
        try:
            if isinstance(usage_info, dict):
                usage_info["anthropic-cache-status"] = cache_status
                usage_info["anthropic-cache-token-credits"] = cache_token_credits
        except Exception:
            pass
        request_time = time.time() - start_time
        
        # The response content is a list of blocks; take only the FINAL text block
        parts = []
        for c in data.get("content", []):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
        text = (parts[-1] if parts else "").strip()

        # Strip accidental code fences if present
        if text.startswith("```") and text.endswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[5:]
            text = text.strip()

        # Include cache header summary in the log line for quick inspection
        if cache_status or cache_token_credits:
            logger.info(
                "LLM Response <- time=%.2fs usage=%s cache_status=%s cache_token_credits=%s",
                request_time, usage_info, cache_status, cache_token_credits,
            )
        else:
            logger.info(f"LLM Response <- time={request_time:.2f}s usage={usage_info}")

        # Optional debug: dump assembled text
        if debug_basepath:
            try:
                with open(f"{debug_basepath}_response_text.txt", "w", encoding="utf-8") as f_txt:
                    f_txt.write(text)
            except Exception:
                logger.debug("Failed to write response text for %s", debug_basepath)
        return text, usage_info

    except requests.exceptions.RequestException as e:
        # Improve error detail: HTTP status/body when available
        status = getattr(getattr(e, "response", None), "status_code", None)
        body = None
        try:
            body = getattr(e, "response", None).text if getattr(e, "response", None) else None
        except Exception:
            body = None
        logger.error(
            "API call failed: %s status=%s body_snippet=%s",
            e, status, (body[:300] + "...") if body else None
        )
        # Optional debug: dump error info
        if debug_basepath:
            try:
                err_info = {
                    "error": str(e),
                    "status": status,
                    "body": body,
                }
                with open(f"{debug_basepath}_error.json", "w", encoding="utf-8") as f_err:
                    json.dump(err_info, f_err, ensure_ascii=False, indent=2)
            except Exception:
                logger.debug("Failed to write error debug for %s", debug_basepath)
        raise


# ------------------------------
# Pause-turn resume helpers (synchronous)
# ------------------------------

def _clean_orphan_tool_use(blocks: List[Any]) -> List[Dict[str, Any]]:
    """Convert SDK blocks to plain dicts and drop a dangling server_tool_use at the end.
    This works around rare cases where a tool_use appears without a matching result.
    """
    logger = logging.getLogger(__name__)
    plain_blocks = _to_json_serializable(blocks)
    if not isinstance(plain_blocks, list):
        return []
    # Ensure dict-like entries
    cleaned: List[Dict[str, Any]] = []
    for b in plain_blocks:
        if isinstance(b, dict):
            cleaned.append(b)
        else:
            # best-effort: wrap as text if unknown
            cleaned.append({"type": "text", "text": str(b)})
    # Determine which tool_use IDs have corresponding results
    result_ids: set = set()
    for blk in cleaned:
        if isinstance(blk, dict) and blk.get("type") in ("web_search_tool_result", "tool_result"):
            tuid = blk.get("tool_use_id")
            if tuid:
                result_ids.add(tuid)

    # Drop any tool_use/server_tool_use blocks that do NOT have a matching result
    # This prevents Anthropic InvalidRequestError during resume when a dangling tool_use is present
    filtered: List[Dict[str, Any]] = []
    seen_tool_use_ids: set = set()
    dropped_orphans: List[str] = []
    dropped_duplicates: List[str] = []
    for blk in cleaned:
        if isinstance(blk, dict) and blk.get("type") in ("server_tool_use", "tool_use"):
            use_id = blk.get("id") or blk.get("tool_use_id")
            # Skip tool_use that has no corresponding result in the provided blocks
            if use_id and use_id not in result_ids:
                dropped_orphans.append(str(use_id))
                continue
            # Also drop exact duplicates to keep history compact
            if use_id and use_id in seen_tool_use_ids:
                dropped_duplicates.append(str(use_id))
                continue
            if use_id:
                seen_tool_use_ids.add(use_id)
        filtered.append(blk)
    # Guardrail logs for visibility
    try:
        if dropped_orphans:
            logger.info("[resume][sanitize] Dropped dangling tool_use ids (no matching tool_result): %s", ", ".join(sorted(set(dropped_orphans))))
        if dropped_duplicates:
            logger.info("[resume][sanitize] Dropped duplicate tool_use ids: %s", ", ".join(sorted(set(dropped_duplicates))))
    except Exception:
        pass
    return filtered


def _resume_until_end_turn(
    api_key: str,
    model: str,
    system_blocks: List[Dict[str, Any]],
    original_user_text: str,
    paused_assistant_blocks: List[Any],
    max_tokens: int,
    use_cache: bool,
    logger: logging.Logger,
    tools: List[Dict[str, Any]],
    thinking: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Resume a pause_turn conversation by re-sending assistant blocks + a 'Please continue' turn.
    Returns (combined_assistant_blocks, usage_totals).
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # Keep beta headers consistent with batch/sync usage: include web_search beta always; add prompt-caching beta when caching is enabled
    headers["anthropic-beta"] = (
        "prompt-caching-2024-07-31, web-search-2025-03-05" if use_cache else "web-search-2025-03-05"
    )

    combined_blocks: List[Dict[str, Any]] = _clean_orphan_tool_use(paused_assistant_blocks)
    usage_totals = {"input_tokens": 0, "output_tokens": 0}

    attempts = 0
    while True:
        # History: original user -> assistant (combined so far) -> user continue
        messages = [
            {"role": "user", "content": original_user_text},
            {"role": "assistant", "content": combined_blocks},
            {"role": "user", "content": "Please continue and return ONLY the final JSON now."},
        ]

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": DEFAULT_TEMPERATURE,
            # IMPORTANT: reuse the original system_blocks object to preserve exact
            # cache_control markers and byte-for-byte identity for prompt caching
            "system": system_blocks,
            "messages": messages,
            "tools": tools,
            "thinking": thinking,
        }

        attempts += 1
        try:
            logger.info("[resume] attempt=%s sending continuation request (messages=%s blocks)", attempts, sum(len(m.get("content", [])) for m in messages if isinstance(m.get("content", []), list)))
        except Exception:
            pass

        try:
            resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            # Log details and break, returning partial combined blocks
            status = getattr(getattr(e, "response", None), "status_code", None)
            body = None
            try:
                body = getattr(e, "response", None).text if getattr(e, "response", None) else None
            except Exception:
                body = None
            logger.error(
                "[resume] request failed: %s status=%s body_snippet=%s",
                e, status, (body[:300] + "...") if body else None
            )
            break
        except ValueError as e:
            # JSON parsing of response failed
            logger.error("[resume] invalid JSON in resume response: %s", e)
            break

        # Track usage
        try:
            usage = data.get("usage", {}) or {}
            usage_totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            usage_totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            logger.info("[resume][usage] in=%s out=%s (cumulative)", usage_totals["input_tokens"], usage_totals["output_tokens"])
        except Exception:
            pass

        # Read stop_reason and content blocks
        stop_reason = data.get("stop_reason")
        new_blocks = data.get("content", []) or []
        if not isinstance(new_blocks, list):
            new_blocks = []

        # Merge into combined (preserve all prior blocks)
        combined_blocks.extend(_clean_orphan_tool_use(new_blocks))

        if stop_reason and stop_reason != "pause_turn":
            logger.info("[resume] reached terminal stop_reason=%s", stop_reason)
            break
        # Continue resuming until a terminal stop_reason is reached
        logger.info("[resume] still paused; continuing loop (attempt=%s)", attempts)

    return combined_blocks, usage_totals


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    # CLI args (only debug for now)
    parser = argparse.ArgumentParser(description="Lipstick QnA generator with web search")
    parser.add_argument("--debug", action="store_true", help="Enable per-product debug dumps under output/debug/")
    parser.add_argument("--no_ingest", action="store_true", help="Disable automatic Pinecone ingestion after writing each JSON output")
    # Batch is ON by default. Use --no_batch to fall back to synchronous single-request mode.
    parser.add_argument("--no_batch", action="store_true", help="Disable Anthropic Message Batches API and use synchronous requests")
    parser.add_argument("--no_sidecars", action="store_true", help="Do not write raw/audit sidecar files alongside the main JSON output")
    parser.add_argument("--no_cache", action="store_true", help="Disable Anthropic prompt caching (use for token price comparison)")
    args, unknown = parser.parse_known_args()
    # Hardcoded file paths
    prompt_path = "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/data/final-prompt-hf.json"
    
    # Data source selection based on USE_EXCEL_DATA parameter
    if USE_EXCEL_DATA:
        input_data_path = "https://kult20256-my.sharepoint.com/:x:/g/personal/harshit_a_kult_app/ER4mU46_r9VAu0XCFYkVzD8Be1E4BFyHPulmXQYVf0ZTtQ?e=H0FrNA"
        logger = logging.getLogger(__name__)
        logger.info("Using Excel data source (online): SharePoint")
    else:
        input_data_path = "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/data/lipstick_list.csv"
        logger = logging.getLogger(__name__)
        logger.info("Using CSV data source (local): %s", input_data_path)
    
    output_dir = "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/output"
    model = DEFAULT_MODEL
    max_tokens = 32000

    # Setup logging
    logger = setup_logging(output_dir)
    # Load environment variables from .env
    try:
        load_dotenv()
        logger.info(".env loaded successfully")
    except Exception as _:
        logger.warning("Failed to load .env; relying on process environment variables")
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint(output_dir)
    
    # Start timing
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("QnA Generation Script Started")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Log checkpoint status
    if checkpoint_data["total_processed"] > 0:
        logger.info("📋 CHECKPOINT STATUS:")
        logger.info(f"   Previously processed: {checkpoint_data['total_processed']} products")
        logger.info(f"   Previously successful: {checkpoint_data['successful']}")
        logger.info(f"   Previously failed: {checkpoint_data['failed']}")
        logger.info(f"   Last updated: {checkpoint_data.get('last_updated', 'Unknown')}")
        logger.info(f"   Completed SKUs: {len(checkpoint_data['completed_skus'])}")
        logger.info(f"   Failed SKUs: {len(checkpoint_data['failed_skus'])}")
        logger.info("   Resuming from checkpoint...")
    else:
        logger.info("📋 Starting fresh - no previous checkpoint found")

    # Select and validate API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY is not set")
        sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    # Determine caching behavior
    use_prompt_cache = not args.no_cache
    logger.info("Prompt caching: %s", "ENABLED" if use_prompt_cache else "DISABLED")
    # DIAGNOSTIC: confirm argv and parsed flags
    try:
        logger.info("argv=%s no_cache=%r use_prompt_cache=%r", sys.argv, args.no_cache, use_prompt_cache)
    except Exception:
        pass

    # Load configuration files
    logger.info("Loading configuration files...")
    try:
        prompt_text = read_prompt(prompt_path)
        logger.info(f"Loaded prompt from: {prompt_path}")
        
        # Always use inline shape-only REQUIRED JSON FORMAT; no external schema file
        
        rows = read_rows(input_data_path)
        logger.info(f"Loaded {len(rows)} products from: {input_data_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration files: {e}")
        sys.exit(1)

    # Filter out already completed products
    remaining_rows = []
    skipped_checkpoint = 0
    
    for row in rows:
        if is_product_completed(row, checkpoint_data):
            skipped_checkpoint += 1
            logger.debug(f"Skipping already completed: {row['brand']} {row['product_name']} - {row['shade_of_lipstick']} (SKU: {row.get('sku', 'N/A')})")
        else:
            remaining_rows.append(row)
    
    if skipped_checkpoint > 0:
        logger.info(f"🔄 Skipped {skipped_checkpoint} already completed products from checkpoint")
    
    logger.info(f"📊 Products to process: {len(remaining_rows)} (out of {len(rows)} total)")

    # Processing statistics for this session
    session_ok, session_fail = 0, 0
    total_input_tokens = 0
    total_output_tokens = 0
    skipped_files = []
    
    if len(remaining_rows) == 0:
        logger.info("🎉 All products already completed! Nothing to process.")
    else:
        logger.info(f"Starting processing of {len(remaining_rows)} remaining products...")
    
    # Prepare debug directory if enabled
    debug_root = os.path.join(output_dir, "debug") if args.debug else None


    # Batch mode: use Anthropic Message Batches API
    if not args.no_batch:
        batch_size = BATCH_SIZE_DEFAULT
        logger.info("Running in BATCH mode (default): batch_size=%s", batch_size)
        ok, fail, batch_in_tokens, batch_out_tokens = run_batch_generation(
            api_key=api_key,
            model=model,
            rows=remaining_rows,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            batch_size=batch_size,
            output_dir=output_dir,
            logger=logger,
            no_ingest=args.no_ingest,
            use_cache=use_prompt_cache,
        )
        session_ok += ok
        session_fail += fail
        total_input_tokens += batch_in_tokens
        total_output_tokens += batch_out_tokens
        # Reload checkpoint to avoid stale stats before final summary
        checkpoint_data = load_checkpoint(output_dir)
        # Skip synchronous loop and proceed to final stats
        total_time = time.time() - start_time
        session_stats = {
            "timestamp": datetime.now().isoformat(),
            "processed": session_ok + session_fail,
            "successful": session_ok,
            "failed": session_fail,
            "duration_seconds": total_time,
            "tokens_used": total_input_tokens + total_output_tokens
        }
        checkpoint_data["session_stats"].append(session_stats)
        save_checkpoint(output_dir, checkpoint_data)
        logger.info("= " * 30)
        logger.info("QnA Generation Session Completed (BATCH mode)")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Session execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info("")
        logger.info("📊 SESSION STATISTICS:")
        logger.info(f"   Products processed this session: {session_ok + session_fail}")
        logger.info(f"   Successful this session: {session_ok}")
        logger.info(f"   Failed this session: {session_fail}")
        logger.info(f"   Skipped (file exists): {len(skipped_files)}")
        logger.info(f"   Skipped (checkpoint): {skipped_checkpoint}")
        logger.info("")
        logger.info("📈 OVERALL STATISTICS:")
        logger.info(f"   Total products in CSV: {len(rows)}")
        logger.info(f"   Total completed: {checkpoint_data['successful']}")
        logger.info(f"   Total failed: {checkpoint_data['failed']}")
        logger.info(f"   Overall success rate: {(checkpoint_data['successful']/(checkpoint_data['successful']+checkpoint_data['failed'])*100):.1f}%" if (checkpoint_data['successful']+checkpoint_data['failed']) > 0 else "N/A")
        logger.info(f"   Completion rate: {(checkpoint_data['successful']/len(rows)*100):.1f}%")
        logger.info("")
        logger.info(f"💰 Token usage this session: {total_input_tokens:,} input + {total_output_tokens:,} output = {total_input_tokens + total_output_tokens:,}")
        logger.info(f"⏱️  Average time per product: {total_time/max(session_ok+session_fail, 1):.2f} seconds")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📋 Checkpoint file: {os.path.join(output_dir, 'checkpoint.json')}")
        logger.info("= " * 30)
        return

    for idx, row in enumerate(remaining_rows, 1):
        product_start_time = time.time()
        product_info = f"{row['brand']} {row['product_name']} - {row['shade_of_lipstick']}"
        if row.get('sku'):
            product_info += f" (SKU: {row['sku']})"
            
        logger.info(f"Processing {idx}/{len(remaining_rows)}: {product_info}")
        
        # Check if file already exists (double-check)
        filename = create_output_filename(row)
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"File already exists: {filename}")
            skipped_files.append(filename)
            # Attempt ingestion for existing files unless disabled
            if not args.no_ingest:
                ing_ok = trigger_ingestion(filepath, logger)
                update_checkpoint(output_dir, row, ing_ok, load_checkpoint(output_dir))
                if not ing_ok:
                    logger.warning("Marked as failed due to ingestion failure for existing file: %s", filename)
            else:
                update_checkpoint(output_dir, row, True, checkpoint_data)
                logger.info("Auto-ingest disabled via --no_ingest; marking existing file as completed: %s", filename)
            continue
        
        # Compute per-product debug base path early (needed in reprocessing block below)
        debug_basepath = None
        if debug_root:
            base_no_ext = os.path.splitext(filename)[0]
            debug_basepath = os.path.join(debug_root, base_no_ext)

        # Reprocessing path: if a raw sidecar exists from a previous failed attempt, try to recover without calling the LLM
        try:
            base_no_ext = os.path.splitext(filename)[0]
            raw_sidecar_path = os.path.join(output_dir, f"{base_no_ext}.raw.json")
            if os.path.exists(raw_sidecar_path):
                logger.info("Found raw sidecar for %s; attempting reprocess without LLM: %s", filename, raw_sidecar_path)
                try:
                    with open(raw_sidecar_path, "r", encoding="utf-8") as f_raw_in:
                        sidecar = json.load(f_raw_in)
                    raw_text = sidecar.get("raw_text", "")
                except Exception as _e:
                    raw_text = ""
                    logger.warning("Failed to read sidecar file %s: %s", raw_sidecar_path, _e)

                if raw_text:
                    # Try to extract and parse JSON from the saved raw LLM text
                    recovered = extract_first_json_object(raw_text) or raw_text
                    try:
                        obj = json.loads(recovered)
                        # Save file
                        with open(filepath, "w", encoding="utf-8") as out:
                            json.dump(obj, out, ensure_ascii=False, indent=2)

                        # Optionally write parsed JSON as well
                        if debug_basepath:
                            try:
                                with open(f"{debug_basepath}_parsed.json", "w", encoding="utf-8") as f_parsed:
                                    json.dump(obj, f_parsed, ensure_ascii=False, indent=2)
                            except Exception:
                                logger.debug("Failed to write parsed JSON for %s", debug_basepath)

                        # Trigger ingestion
                        ing_ok = True
                        if not args.no_ingest:
                            ing_ok = trigger_ingestion(filepath, logger)
                        else:
                            logger.info("Auto-ingest disabled via --no_ingest; skipping ingestion for %s", filepath)

                        # Update checkpoint and continue to next product
                        update_checkpoint(output_dir, row, ing_ok, load_checkpoint(output_dir))
                        if not ing_ok:
                            logger.warning("Marked as failed due to ingestion failure (reprocessed): %s", filename)
                        logger.info("✓ Row %s (reprocessed): SUCCESS -> %s", idx, filename)
                        continue
                    except json.JSONDecodeError:
                        logger.info("Reprocess failed to parse sidecar raw_text; proceeding to call LLM for %s", filename)
                else:
                    logger.info("Sidecar present but no raw_text found; proceeding to call LLM for %s", filename)
        except Exception as _e:
            logger.warning("Unexpected error during sidecar reprocessing path for %s: %s", filename, _e)

        system_text, user_msg = build_user_message(prompt_text, row, use_natural_generation=True)
        
        try:
            raw, usage_info = call_claude_with_web_search(
                api_key,
                model,
                user_msg,
                max_tokens=max_tokens,
                debug_basepath=debug_basepath,
                system_text=system_text,
                use_cache=use_prompt_cache,
            )

            # Pre-clean: many providers wrap JSON in ```json fences; extract first JSON block if present
            raw_for_parse = extract_first_json_object(raw) or raw

            try:
                obj = json.loads(raw_for_parse)
            except json.JSONDecodeError:
                # Attempt to extract JSON object if assistant added prose before/after
                cleaned = extract_first_json_object(raw)
                if cleaned is None:
                    raise
                # Optionally write the cleaned JSON to debug
                if debug_basepath:
                    try:
                        with open(f"{debug_basepath}_response_text.cleaned.json", "w", encoding="utf-8") as f_clean:
                            f_clean.write(cleaned)
                    except Exception:
                        logger.debug("Failed to write cleaned JSON for %s", debug_basepath)
                obj = json.loads(cleaned)
            
            # Track token usage
            input_tokens = usage_info.get("input_tokens", 0)
            output_tokens = usage_info.get("output_tokens", 0)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            # Save file
            with open(filepath, "w", encoding="utf-8") as out:
                json.dump(obj, out, ensure_ascii=False, indent=2)

            # Optionally write raw-response and/or audit file (disabled by default)
            if WRITE_RAW_FULL or WRITE_AUDIT_FILES:
                try:
                    base_no_ext = os.path.splitext(filename)[0]
                    if WRITE_RAW_FULL:
                        # Full raw text from the LLM
                        raw_full_txt_path = os.path.join(output_dir, f"{base_no_ext}.raw.full.txt")
                        with open(raw_full_txt_path, "w", encoding="utf-8") as f_raw_full:
                            f_raw_full.write(raw)
                    if WRITE_AUDIT_FILES:
                        # Audit JSON comparing raw chars vs parsed JSON chars
                        parsed_json_str = json.dumps(obj, ensure_ascii=False)
                        audit = {
                            "filename": filename,
                            "raw_chars": len(raw or ""),
                            "parsed_json_chars": len(parsed_json_str),
                            "dropped_chars": max(len(raw or "") - len(parsed_json_str), 0),
                            "timestamp": datetime.now().isoformat(),
                        }
                        audit_path = os.path.join(output_dir, f"{base_no_ext}.audit.json")
                        with open(audit_path, "w", encoding="utf-8") as f_audit:
                            json.dump(audit, f_audit, ensure_ascii=False, indent=2)
                except Exception as _e:
                    logger.debug("Failed to write optional raw/audit files for %s: %s", filename, _e)

            # Optional debug: save parsed JSON as well (duplicate of output for quick diff)
            if debug_basepath:
                try:
                    with open(f"{debug_basepath}_parsed.json", "w", encoding="utf-8") as f_parsed:
                        json.dump(obj, f_parsed, ensure_ascii=False, indent=2)
                except Exception:
                    logger.debug("Failed to write parsed JSON for %s", debug_basepath)

            # Trigger ingestion
            ing_ok = True
            if not args.no_ingest:
                ing_ok = trigger_ingestion(filepath, logger)
            else:
                logger.info("Auto-ingest disabled via --no_ingest; skipping ingestion for %s", filepath)
             
            product_time = time.time() - product_start_time
            session_ok += 1
            
            # Update checkpoint
            update_checkpoint(output_dir, row, ing_ok, load_checkpoint(output_dir))
            if not ing_ok:
                logger.warning("Marked as failed due to ingestion failure: %s", filename)
            
            logger.info(f"✓ Row {idx}: SUCCESS -> {filename} (Time: {product_time:.2f}s, Tokens: {input_tokens}→{output_tokens})")
            
        except requests.HTTPError as http_e:
            logger.error(f"✗ Row {idx}: HTTP error: {http_e}")
            # Debug error already written inside call if last attempt failed
            session_fail += 1
            update_checkpoint(output_dir, row, False, checkpoint_data)
            continue
        except json.JSONDecodeError as jd:
            logger.error(f"✗ Row {idx}: JSON decode error: {jd}")
            logger.debug(f"Raw response: {raw[:500]}...")
            # Persist raw LLM output to a sidecar file so we don't lose the response
            try:
                base_no_ext = os.path.splitext(filename)[0]
                raw_sidecar_path = os.path.join(output_dir, f"{base_no_ext}.raw.json")
                sidecar_payload = {
                    "raw_text": raw,
                    "error": str(jd),
                    "model": model,
                    "product": {
                        "brand": row.get("brand", ""),
                        "product_name": row.get("product_name", ""),
                        "shade_of_lipstick": row.get("shade_of_lipstick", ""),
                        "sku": row.get("sku", "")
                    },
                    "timestamp": datetime.now().isoformat()
                }
                with open(raw_sidecar_path, "w", encoding="utf-8") as f_raw_out:
                    json.dump(sidecar_payload, f_raw_out, ensure_ascii=False, indent=2)
                logger.info("Saved raw LLM output to %s", raw_sidecar_path)
            except Exception as _e:
                logger.warning("Failed to save raw sidecar file: %s", _e)
            if debug_basepath:
                try:
                    with open(f"{debug_basepath}_json_error.txt", "w", encoding="utf-8") as f_jerr:
                        f_jerr.write(str(jd) + "\n\n")
                        f_jerr.write(raw)
                except Exception:
                    logger.debug("Failed to write JSON error file for %s", debug_basepath)
            session_fail += 1
            update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
            continue
        except Exception as e:
            logger.error(f"✗ Row {idx}: Unexpected error: {e}")
            if debug_basepath:
                try:
                    with open(f"{debug_basepath}_unexpected_error.txt", "w", encoding="utf-8") as f_uerr:
                        f_uerr.write(str(e))
                except Exception:
                    logger.debug("Failed to write unexpected error file for %s", debug_basepath)
            session_fail += 1
            update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
            continue

    # Final statistics
    total_time = time.time() - start_time
    
    # Update session stats in checkpoint
    session_stats = {
        "timestamp": datetime.now().isoformat(),
        "processed": session_ok + session_fail,
        "successful": session_ok,
        "failed": session_fail,
        "duration_seconds": total_time,
        "tokens_used": total_input_tokens + total_output_tokens
    }
    checkpoint_data["session_stats"].append(session_stats)
    save_checkpoint(output_dir, checkpoint_data)
    
    logger.info("=" * 60)
    logger.info("QnA Generation Session Completed")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Session execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("")
    logger.info("📊 SESSION STATISTICS:")
    logger.info(f"   Products processed this session: {session_ok + session_fail}")
    logger.info(f"   Successful this session: {session_ok}")
    logger.info(f"   Failed this session: {session_fail}")
    logger.info(f"   Skipped (file exists): {len(skipped_files)}")
    logger.info(f"   Skipped (checkpoint): {skipped_checkpoint}")
    logger.info("")
    logger.info("📈 OVERALL STATISTICS:")
    logger.info(f"   Total products in CSV: {len(rows)}")
    logger.info(f"   Total completed: {checkpoint_data['successful']}")
    logger.info(f"   Total failed: {checkpoint_data['failed']}")
    logger.info(f"   Overall success rate: {(checkpoint_data['successful']/(checkpoint_data['successful']+checkpoint_data['failed'])*100):.1f}%" if (checkpoint_data['successful']+checkpoint_data['failed']) > 0 else "N/A")
    logger.info(f"   Completion rate: {(checkpoint_data['successful']/len(rows)*100):.1f}%")
    logger.info("")
    logger.info(f"💰 Token usage this session: {total_input_tokens:,} input + {total_output_tokens:,} output = {total_input_tokens + total_output_tokens:,}")
    logger.info(f"⏱️  Average time per product: {total_time/max(session_ok+session_fail, 1):.2f} seconds")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📋 Checkpoint file: {os.path.join(output_dir, 'checkpoint.json')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()