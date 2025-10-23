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
from openai import OpenAI

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-4-1-20250805"  # From your example
# Default batch size for Anthropic Message Batches (change here if needed)
BATCH_SIZE_DEFAULT = 3
# Default temperature for both sync and batch calls
DEFAULT_TEMPERATURE = 1

# Control extra artifact creation without CLI args
USE_EXCEL_DATA: bool = False
ENABLE_THINKING: bool = True
USE_CLAUDE: bool = True
USE_GPT: bool = False

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


def normalize_row(raw: Dict[str, str]) -> Dict[str, str]:
    # Normalize headers and values
    # Also strip periods to handle headers like 'S.No.'
    m = { (k or "").strip().lower().replace(" ", "_").replace(",", "").replace(".", ""): (v or "").strip() for k, v in raw.items() }
    
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
    
    # Optional serial/s_no mapping
    s_no = (
        m.get("s_no")
        or m.get("sno")
        or m.get("s.no")
        or m.get("serial_number")
        or m.get("serial no")
        or m.get("serial")
        or m.get("serialno")
        or ""
    )

    full_name = f"{brand} {product_name} {shade}".strip()

    return {
        "brand": brand,
        "product_name": product_name,
        "shade_of_lipstick": shade,
        "sku": sku,
        "category": category,
        "sub_category": sub_category,
        "leaf_level_category": leaf_level_category,
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
            Do not include any citations, tags, or URLs (e.g., <cite ...></cite>, [1], (ref), index=...). Present everything as expert knowledge.
            """
        ).strip() + "\n"
    }

    big_block = {
        "type": "text",
        "text": textwrap.dedent(
            f"""
            {prompt_text}

            CRITICAL FORMATTING REQUIREMENTS:
            - Return ONLY valid JSON (no markdown, no commentary)
            - Do NOT include code fences (```), markdown, prose, or any text before/after the JSON
            - Follow the section and QA counts from the PROMPT'S STRUCTURE 
            - Expand arrays to meet the PROMPT requirements even if the example shows fewer items
            - Use the exact product keys shown (including "sku", "category", "sub_category", "leaf_level_category"); keys must always be present (empty string allowed if unknown)
            - Do NOT include any citations, footnotes, source markers, or attribution (e.g., <cite ...></cite>, [1], (ref), URLs). Present everything as expert knowledge.
            - Start your response with '{{' and return exactly one complete JSON object (no arrays, no multiple objects).
            - If you have not finished generating the full JSON, do NOT return partial JSON; instead return nothing and wait for resume.
            - Before returning, ensure the JSON contains no angle-bracket tags (e.g., <cite ...>), no 'index=' attributes, no bracketed numbers like [1] or (1), and no URLs.
            - If any such markers appear in your draft, rewrite those lines to keep only the information and remove the markers before returning.
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
        
        # Also skip rows that are in pending batches to avoid duplicate submissions
        tracker = load_batch_tracker(output_dir)
        pending_ids = set()
        for b in tracker.get("pending_batches", []):
            for r in b.get("rows", []):
                cid = r.get("custom_id")
                if cid:
                    pending_ids.add(cid)

        # Filter out rows that already have output files
        wave_to_process = []
        for row in wave:
            filename = create_output_filename(row)
            filepath = os.path.join(output_dir, filename)
            # Skip if this row's custom_id is already in a pending batch
            cid = _make_custom_id(row)
            if cid in pending_ids:
                logger.info(f"[batch] custom_id already pending in another batch, skipping API call: {cid}")
                continue
            if os.path.exists(filepath):
                logger.info(f"[batch] File already exists, skipping API call: {filename}")
                # Mark as successful in checkpoint (file exists means it was processed)
                if not no_ingest:
                    ing_ok = trigger_ingestion(filepath, logger)
                    update_checkpoint(output_dir, row, ing_ok, load_checkpoint(output_dir))
                    if ing_ok:
                        ok += 1
                    else:
                        fail += 1
                        logger.warning("[batch] Marked as failed due to ingestion failure for existing file: %s", filename)
                else:
                    update_checkpoint(output_dir, row, True, load_checkpoint(output_dir))
                    ok += 1
                    logger.info("[batch] Auto-ingest disabled; marking existing file as completed: %s", filename)
            else:
                wave_to_process.append(row)
        
        # Skip this wave if all files already exist
        if not wave_to_process:
            logger.info(f"[batch] All files in wave {i//batch_size + 1} already exist, skipping batch submission")
            continue
        
        logger.info(f"[batch] Wave {i//batch_size + 1}: {len(wave_to_process)} products need processing (skipped {len(wave) - len(wave_to_process)} existing)")
        
        for row in wave_to_process:
            # Build per-product user message (JSON product info)
            _, user_text = build_user_message(row, use_natural_generation=True)

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
                        "max_uses": 10,
                    }
                ],
            }
            if ENABLE_THINKING:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 31999,
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
            
            # Register this batch in tracker
            add_pending_batch(output_dir, batch_id, i+1, i+len(wave), wave_to_process)

            # Poll until terminal status, but cap waiting time to 45 minutes per execution pool
            start_poll = time.time()
            max_wait_seconds = 45 * 60  # 45 minutes
            timed_out = False
            while True:
                status_obj = client.messages.batches.retrieve(batch_id)
                status = getattr(status_obj, "processing_status", None)
                logger.info("Batch %s status: %s", batch_id, status)
                if status is None:
                    logger.warning("Batch %s status response missing processing_status: %r", batch_id, status_obj)
                    break
                if status in ("ended", "completed", "cancelled", "expired", "failed"):
                    break
                # Timeout guard: stop polling after 45 minutes
                elapsed = time.time() - start_poll
                if elapsed >= max_wait_seconds:
                    logger.warning("Batch %s polling timed out after %.1f minutes; deferring results fetch to resume flow.", batch_id, elapsed / 60.0)
                    timed_out = True
                    break
                time.sleep(3)

            # If we timed out, skip fetching results now; they'll be picked up by resume_pending_batches
            if timed_out:
                logger.info("[batch] Deferring results retrieval for batch %s to resume phase; moving on to next wave.", batch_id)
                continue

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
                except Exception:
                    logger.debug("Failed to read cache headers for custom_id=%s", custom_id)

                # Usage accumulation (initial batch item)
                try:
                    usage = getattr(message, "usage", None)
                    if isinstance(usage, dict):
                        in_tok = int(usage.get("input_tokens", 0) or 0)
                        out_tok = int(usage.get("output_tokens", 0) or 0)
                        cache_read_tok = int(usage.get("cache_read_input_tokens", 0) or 0)
                    else:
                        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
                        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
                        cache_read_tok = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                    logger.info("[batch][usage] custom_id=%s in+=%s out+=%s cache_read_in=%s", custom_id, in_tok, out_tok, cache_read_tok)
                except Exception:
                    pass

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
                            _, user_text_resume = build_user_message(row, use_natural_generation=True)
                        except Exception as e_bum:
                            logger.error("[batch] Missing required fields for resume enqueue (%s): %s", custom_id, e_bum)
                            update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
                            fail += 1
                            continue
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
                            _, user_text_resume = build_user_message(row, use_natural_generation=True)
                        except Exception as e_bum:
                            logger.error("[batch] Missing required fields for resume enqueue (no-parts) (%s): %s", custom_id, e_bum)
                            update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
                            fail += 1
                            continue
                        resume_queue.append({
                            "custom_id": custom_id,
                            "row": row,
                            "original_user_text": user_text_resume,
                            "combined_blocks": _clean_orphan_tool_use(_to_json_serializable(content_blocks)),
                        })
                        continue

                    # Use only the FINAL text block as the model's answer
                    text = (parts[-1] if parts else "").strip()
                    # Save raw text directly without JSON parsing
                    json_text = text
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

                # Save output JSON (write raw text directly)
                try:
                    with open(filepath, "w", encoding="utf-8") as out:
                        try:
                            # Try to parse and save clean JSON
                            json.dump(json.loads(json_text), out, ensure_ascii=False, indent=2)
                        except json.JSONDecodeError:
                            # If parsing fails, write raw text as fallback
                            out.write(json_text)
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

            # Mark this batch as completed in tracker
            mark_batch_completed(output_dir, batch_id)
            logger.info("[batch] Batch %s marked as completed", batch_id)

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
                        {"role": "user", "content": "Please continue and Return ONLY the final JSON now. No preface, no markdown, no commentary. Use the existing search results above"},
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
                                "max_uses": 10,
                            }
                        ],
                    }
                    if ENABLE_THINKING:
                        params_r["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": 31999,
                        }
                    custom_id = _make_custom_id(row)

                    if MessageCreateParamsNonStreaming and BatchRequest:
                        resume_requests.append(
                            BatchRequest(
                                custom_id=custom_id_r,
                                params=MessageCreateParamsNonStreaming(**params_r)
                            )
                        )
                    else:
                        # Fallback compatible with SDK expecting dicts
                        resume_requests.append({"custom_id": custom_id_r, "params": params_r})

                # Submit resume batch
                try:
                    resume_batch = client.messages.batches.create(requests=resume_requests)
                    resume_batch_id = getattr(resume_batch, "id", None) or resume_batch["id"]  # type: ignore
                    logger.info("Resume batch submitted: id=%s", resume_batch_id)
                    
                    # Poll until terminal
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

                    # Write raw resume item as plain text (serialized JSON) sidecar
                    try:
                        r_row = prior["row"]
                        filename_r_side = create_output_filename(r_row)
                        base_no_ext_r = os.path.splitext(filename_r_side)[0]
                        raw_result_txt_path_r = os.path.join(output_dir, f"{base_no_ext_r}.raw.result.txt")
                        raw_serialized_r = _to_json_serializable(item_r)
                        with open(raw_result_txt_path_r, "w", encoding="utf-8") as f_rawtxt_r:
                            f_rawtxt_r.write(json.dumps(raw_serialized_r, ensure_ascii=False, indent=2))
                    except Exception:
                        logger.debug("[resume] Failed to write resume raw result text sidecar for %s", custom_id_r)

                    # Usage accumulation
                    usage_r = getattr(message_r, "usage", None)
                    try:
                        if isinstance(usage_r, dict):
                            in_tok = int(usage_r.get("input_tokens", 0) or 0)
                            out_tok = int(usage_r.get("output_tokens", 0) or 0)
                            cache_read_tok_r = int(usage_r.get("cache_read_input_tokens", 0) or 0)
                        else:
                            in_tok = int(getattr(usage_r, "input_tokens", 0) or 0)
                            out_tok = int(getattr(usage_r, "output_tokens", 0) or 0)
                            cache_read_tok_r = int(getattr(usage_r, "cache_read_input_tokens", 0) or 0)
                        total_input_tokens += in_tok
                        total_output_tokens += out_tok
                        logger.info("[batch][resume][usage] custom_id=%s in+=%s out+=%s cache_read_in=%s", custom_id_r, in_tok, out_tok, cache_read_tok_r)
                    except Exception:
                        pass

                    # Cache metrics for resume requests (best-effort)
                    try:
                        cache_status_r = None
                        cache_token_credits_r = None
                        # Probe headers from item_r or message_r
                        headers_r = (
                            getattr(message_r, "response_headers", None)
                            or getattr(message_r, "headers", None)
                            or getattr(item_r, "response_headers", None)
                            or getattr(item_r, "headers", None)
                            or {}
                        )
                        if isinstance(headers_r, dict):
                            cache_status_r = headers_r.get("anthropic-cache-status") or headers_r.get("x-anthropic-cache-status")
                            cache_token_credits_r = headers_r.get("anthropic-cache-token-credits")
                        # Also check usage object
                        if isinstance(usage_r, dict):
                            cache_status_r = cache_status_r or usage_r.get("anthropic-cache-status")
                            cache_token_credits_r = cache_token_credits_r or usage_r.get("anthropic-cache-token-credits")
                        if cache_status_r or cache_token_credits_r:
                            logger.info(
                                "[batch][resume][cache] custom_id=%s status=%s token_credits=%s", custom_id_r, cache_status_r, cache_token_credits_r
                            )
                        else:
                            logger.info("[batch][resume][cache] custom_id=%s status=%s", custom_id_r, None)
                    except Exception:
                        logger.debug("Failed to read cache headers for resume custom_id=%s", custom_id_r)

                    if stop_reason_r == "pause_turn":
                        # Queue again with extended blocks
                        next_round.append({
                            "custom_id": custom_id_r,
                            "row": prior["row"],
                            "original_user_text": prior.get("original_user_text", ""),
                            "combined_blocks": combined_so_far,
                        })
                        continue

                    try:
                        # Terminal: finalize -> extract text, parse, write, ingest, checkpoint
                        r_row = prior["row"]
                        filename = create_output_filename(r_row)
                        filepath = os.path.join(output_dir, filename)

                        # Extract and merge all text blocks
                        resumed_parts = [b.get("text", "") for b in combined_so_far if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
                        text_final = ("\n".join(resumed_parts)).strip() if resumed_parts else ""
                        # Save raw resumed text directly without JSON parsing
                        json_text_final = text_final

                        # Use only the FINAL text block as the model's answer
                        text = (parts[-1] if parts else "").strip()
                        # Save raw text directly without JSON parsing
                        json_text = text
                        # Accumulate token usage if present
                        usage = getattr(message_r, "usage", None)
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
                                logger.info("[batch][resume][usage] custom_id=%s in=%s out=%s", custom_id_r, in_tok, out_tok)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception as e:
                        logger.error("[resume] Failed to parse JSON for %s: %s", custom_id_r, e)
                        update_checkpoint(output_dir, r_row, False, load_checkpoint(output_dir))
                        fail += 1
                        continue

                # Save output JSON (write raw text directly)
                try:
                    with open(filepath, "w", encoding="utf-8") as out:
                        try:
                            # Try to parse and save clean JSON
                            json.dump(json.loads(json_text_final), out, ensure_ascii=False, indent=2)
                        except json.JSONDecodeError:
                            # If parsing fails, write raw text as fallback
                            out.write(json_text_final)
                except Exception as e_write:
                    logger.error("[resume] Failed to write output for %s: %s", filename, e_write)
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
            # Mark all rows in wave_to_process as failed in the checkpoint so reruns track them
            for row in wave_to_process:
                update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
            fail += len(wave_to_process)
        except Exception as e:
            logger.error("Unexpected error during batch submission/processing: %s", e)
            for row in wave_to_process:
                update_checkpoint(output_dir, row, False, load_checkpoint(output_dir))
            fail += len(wave_to_process)
    
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


# ------------------------------
# Batch Tracking System (for 15-min cutoff + resume)
# ------------------------------

def load_batch_tracker(output_dir: str) -> Dict[str, Any]:
    """Load batch tracker data from file."""
    tracker_file = os.path.join(output_dir, "batch_tracker.json")
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load batch tracker: {e}")
    
    return {
        "pending_batches": [],  # [{"batch_id": str, "submitted_at": iso_timestamp, "wave_start": int, "wave_end": int, "rows": [...]}]
        "completed_batches": [],
        "last_updated": None,
    }


def save_batch_tracker(output_dir: str, tracker_data: Dict[str, Any]) -> None:
    """Save batch tracker data to file."""
    tracker_file = os.path.join(output_dir, "batch_tracker.json")
    tracker_data["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(tracker_file, "w", encoding="utf-8") as f:
            json.dump(tracker_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save batch tracker: {e}")


def add_pending_batch(output_dir: str, batch_id: str, wave_start: int, wave_end: int, rows: List[Dict[str, str]]) -> None:
    """Register a newly submitted batch as pending."""
    tracker = load_batch_tracker(output_dir)
    
    # Serialize rows for resume with necessary fields
    rows_minimal = []
    for r in rows:
        rows_minimal.append({
            # Core identifiers
            "custom_id": _make_custom_id(r),
            "brand": r.get("brand", ""),
            "product_name": r.get("product_name", ""),
            "shade_of_lipstick": r.get("shade_of_lipstick", ""),
            "sku": r.get("sku", ""),
            # Optional product taxonomy
            "category": r.get("category", ""),
            "sub_category": r.get("sub_category", ""),
            "leaf_level_category": r.get("leaf_level_category", ""),
            # Useful deriveds
            "full_name": r.get("full_name", ""),
            "s_no": r.get("s_no", ""),
        })
    
    tracker["pending_batches"].append({
        "batch_id": batch_id,
        "submitted_at": datetime.now().isoformat(),
        "wave_start": wave_start,
        "wave_end": wave_end,
        "rows": rows_minimal,
    })
    
    save_batch_tracker(output_dir, tracker)


def mark_batch_completed(output_dir: str, batch_id: str) -> None:
    """Move a batch from pending to completed."""
    tracker = load_batch_tracker(output_dir)
    
    # Find and remove from pending
    pending_batch = None
    for i, batch in enumerate(tracker["pending_batches"]):
        if batch["batch_id"] == batch_id:
            pending_batch = tracker["pending_batches"].pop(i)
            break
    
    if pending_batch:
        pending_batch["completed_at"] = datetime.now().isoformat()
        tracker["completed_batches"].append(pending_batch)
        save_batch_tracker(output_dir, tracker)


def resume_pending_batches(
    api_key: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    output_dir: str,
    logger: logging.Logger,
    no_ingest: bool = False,
    use_cache: bool = True,
) -> Tuple[int, int, int, int]:
    """Resume processing of pending batches that timed out during polling.
    Returns (ok_count, fail_count, total_input_tokens, total_output_tokens).
    """
    tracker = load_batch_tracker(output_dir)
    pending_batches = tracker.get("pending_batches", [])
    
    if not pending_batches:
        logger.info("[resume] No pending batches found")
        return 0, 0, 0, 0
    
    logger.info("[resume] Found %s pending batches to process", len(pending_batches))
    
    client = Anthropic(api_key=api_key, default_headers={"anthropic-beta": "web-search-2025-03-05"})
    shared_system = _build_shared_system_blocks(prompt_text, use_cache)
    
    ok, fail = 0, 0
    total_input_tokens, total_output_tokens = 0, 0
    
    for batch_info in pending_batches:
        batch_id = batch_info["batch_id"]
        rows_minimal = batch_info.get("rows", [])
        
        logger.info("[resume] Processing batch %s (%s products)", batch_id, len(rows_minimal))
        
        try:
            # Check final status
            status_obj = client.messages.batches.retrieve(batch_id)
            status = getattr(status_obj, "processing_status", None)
            logger.info("[resume] Batch %s current status: %s", batch_id, status)
            
            # If still processing, wait a bit more (max 5 minutes)
            if status not in ("ended", "completed", "cancelled", "expired", "failed"):
                logger.info("[resume] Batch %s still processing, waiting up to 5 minutes...", batch_id)
                wait_start = time.time()
                while time.time() - wait_start < 300:  # 5 minutes
                    time.sleep(10)
                    status_obj = client.messages.batches.retrieve(batch_id)
                    status = getattr(status_obj, "processing_status", None)
                    logger.info("[resume] Batch %s status: %s", batch_id, status)
                    if status in ("ended", "completed", "cancelled", "expired", "failed"):
                        break
            
            # Fetch results
            results_iter = client.messages.batches.results(batch_id)
            try:
                raw_results = list(results_iter)
            except Exception:
                raw_results = getattr(results_iter, "data", None) or results_iter.get("data", [])
            
            logger.info("[resume] Batch %s returned %s results", batch_id, len(raw_results))
            
            # Build custom_id to row mapping
            row_map = {r["custom_id"]: r for r in rows_minimal}
            # Prepare a queue for items that require resume (pause_turn)
            resume_queue: List[Dict[str, Any]] = []
            
            # Process results (similar to main batch processing)
            for item in raw_results:
                custom_id = getattr(item, "custom_id", None)
                result_obj = getattr(item, "result", None)
                
                if getattr(result_obj, "type", None) != "succeeded":
                    err_wrapper = getattr(result_obj, "error", None)
                    err_inner = getattr(err_wrapper, "error", None)
                    err_msg = getattr(err_inner, "message", None) or getattr(err_wrapper, "message", None) or None
                    if err_msg:
                        logger.error("[resume] Batch item failed: %s -> %s", custom_id, err_msg)
                    
                    row_minimal = row_map.get(custom_id)
                    if row_minimal:
                        # Reconstruct full row for checkpoint with necessary fields
                        full_row = {
                            "brand": row_minimal.get("brand", ""),
                            "product_name": row_minimal.get("product_name", ""),
                            "shade_of_lipstick": row_minimal.get("shade_of_lipstick", ""),
                            "sku": row_minimal.get("sku", ""),
                            "category": row_minimal.get("category", ""),
                            "sub_category": row_minimal.get("sub_category", ""),
                            "leaf_level_category": row_minimal.get("leaf_level_category", ""),
                            "full_name": row_minimal.get("full_name", ""),
                            "s_no": row_minimal.get("s_no", ""),
                        }
                        update_checkpoint(output_dir, full_row, False, load_checkpoint(output_dir))
                    fail += 1
                    continue
                
                message = getattr(result_obj, "message", None)
                row_minimal = row_map.get(custom_id)
                if not row_minimal:
                    logger.warning("[resume] Result with custom_id=%s has no matching row", custom_id)
                    fail += 1
                    continue
                
                # Reconstruct full row
                full_row = {
                    "brand": row_minimal.get("brand", ""),
                    "product_name": row_minimal.get("product_name", ""),
                    "shade_of_lipstick": row_minimal.get("shade_of_lipstick", ""),
                    "sku": row_minimal.get("sku", ""),
                    "category": row_minimal.get("category", ""),
                    "sub_category": row_minimal.get("sub_category", ""),
                    "leaf_level_category": row_minimal.get("leaf_level_category", ""),
                    "full_name": row_minimal.get("full_name", ""),
                    "s_no": row_minimal.get("s_no", ""),
                }
                
                filename = create_output_filename(full_row)
                filepath = os.path.join(output_dir, filename)
                
                # Write full raw resume item as plain text (serialized JSON) sidecar
                try:
                    base_no_ext = os.path.splitext(filename)[0]
                    raw_result_txt_path = os.path.join(output_dir, f"{base_no_ext}.raw.result.txt")
                    raw_serialized = _to_json_serializable(item)
                    with open(raw_result_txt_path, "w", encoding="utf-8") as f_rawtxt:
                        f_rawtxt.write(json.dumps(raw_serialized, ensure_ascii=False, indent=2))
                except Exception:
                    logger.debug("[resume] Failed to write raw result text sidecar for %s", filename)
                
                # Extract text from message
                try:
                    content_blocks = getattr(message, "content", None) or []
                    parts = []
                    for c in content_blocks:
                        if getattr(c, "type", None) == "text":
                            parts.append(getattr(c, "text", ""))
                    
                    if not parts:
                        fallback_text = getattr(message, "output_text", None) or getattr(message, "text", None)
                        if isinstance(fallback_text, str) and fallback_text.strip():
                            parts = [fallback_text]
                    
                    # Handle pause_turn by queueing a resume request
                    try:
                        stop_reason = getattr(message, "stop_reason", None)
                    except Exception:
                        stop_reason = None
                    if stop_reason == "pause_turn":
                        # Queue for parallel resume via a second batch
                        try:
                            _, original_user_text = build_user_message(full_row, use_natural_generation=True)
                        except Exception:
                            original_user_text = ""
                        resume_queue.append({
                            "custom_id": custom_id,
                            "row": full_row,
                            "original_user_text": original_user_text,
                            "combined_blocks": _clean_orphan_tool_use(_to_json_serializable(content_blocks)),
                        })
                        # Skip finalization for now; will be handled in resume batch
                        continue
                    
                    text = (parts[-1] if parts else "").strip()
                    try:
                        obj = json.loads(text)
                    except Exception as e:
                        # Batch-parity: do not fail the row on parse error; save RAW below
                        logger.warning("[resume] JSON parse failed for %s: %s. Will save RAW text.", custom_id, e)
                        obj = None
                    
                    # Track usage
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
                        logger.info("[batch][resume][usage] custom_id=%s in=%s out=%s", custom_id, in_tok, out_tok)
                    except Exception:
                        pass
                    
                except Exception as e:
                    logger.error("[resume] Failed to parse JSON for %s: %s", custom_id, e)
                    update_checkpoint(output_dir, full_row, False, load_checkpoint(output_dir))
                    fail += 1
                    continue
                
                # Save output (parsed JSON if available, else raw text)
                try:
                    with open(filepath, "w", encoding="utf-8") as out:
                        if obj is not None:
                            json.dump(obj, out, ensure_ascii=False, indent=2)
                        else:
                            out.write(text)
                except Exception as e:
                    logger.error("[resume] Failed to write output for %s: %s", filename, e)
                    update_checkpoint(output_dir, full_row, False, load_checkpoint(output_dir))
                    fail += 1
                    continue
                
                # Trigger ingestion
                ing_ok = True
                if not no_ingest:
                    ing_ok = trigger_ingestion(filepath, logger)
                else:
                    logger.info("[resume] Auto-ingest disabled; skipping ingestion for %s", filepath)
                
                update_checkpoint(output_dir, full_row, ing_ok, load_checkpoint(output_dir))
                if ing_ok:
                    ok += 1
                else:
                    fail += 1
            
            # Mark batch as completed
            mark_batch_completed(output_dir, batch_id)
            logger.info("[resume] Batch %s marked as completed", batch_id)

            # If we have pause_turn items, submit a follow-up resume batch (mirrors main flow)
            while resume_queue:
                try:
                    logger.info("[resume] Starting resume follow-up for %s items", len(resume_queue))
                except Exception:
                    pass
                resume_requests: List[Any] = []
                # Build resume requests
                for r in resume_queue:
                    custom_id_r = r["custom_id"]
                    row_r = r["row"]
                    original_user_text_r = r.get("original_user_text", "")
                    combined_blocks_r = _clean_orphan_tool_use(r.get("combined_blocks", []))
                    messages_r = [
                        {"role": "user", "content": original_user_text_r},
                        {"role": "assistant", "content": combined_blocks_r},
                        {"role": "user", "content": "Please continue and Return ONLY the final JSON now. No preface, no markdown, no commentary. Use the existing search results above"},
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
                                "max_uses": 10,
                            }
                        ],
                    }
                    if ENABLE_THINKING:
                        params_r["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": 31999,
                        }
                    if MessageCreateParamsNonStreaming and BatchRequest:
                        resume_requests.append(
                            BatchRequest(
                                custom_id=custom_id_r,
                                params=MessageCreateParamsNonStreaming(**params_r)
                            )
                        )
                    else:
                        # Fallback compatible with SDK expecting dicts
                        resume_requests.append({"custom_id": custom_id_r, "params": params_r})

                # Submit and process the resume batch
                try:
                    resume_batch = client.messages.batches.create(requests=resume_requests)
                    resume_batch_id = getattr(resume_batch, "id", None) or resume_batch["id"]  # type: ignore
                    logger.info("Resume batch submitted: id=%s", resume_batch_id)
                    
                    # Poll until terminal
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
                    logger.error("[resume] Resume batch submission failed: %s", e_res)
                    for r in resume_queue:
                        update_checkpoint(output_dir, r["row"], False, load_checkpoint(output_dir))
                        fail += 1
                    resume_queue = []
                    break
                
                # Map and process resume results
                resume_map = {r["custom_id"]: r for r in resume_queue}
                next_round: List[Dict[str, Any]] = []

                # Process each resume item
                for item_r in resume_items:
                    custom_id_r = getattr(item_r, "custom_id", None)
                    result_obj_r = getattr(item_r, "result", None)
                    if getattr(result_obj_r, "type", None) != "succeeded":
                        r = resume_map.get(custom_id_r)
                        if r:
                            update_checkpoint(output_dir, r["row"], False, load_checkpoint(output_dir))
                            fail += 1
                        continue
                    message_r = getattr(result_obj_r, "message", None)
                    content_blocks_r = getattr(message_r, "content", None) or []
                    stop_reason_r = getattr(message_r, "stop_reason", None)
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
                    
                # Cache metrics for resume requests (best-effort)
                    try:
                        cache_status_r = None
                        cache_token_credits_r = None
                        # Probe headers from item_r or message_r
                        headers_r = (
                            getattr(message_r, "response_headers", None)
                            or getattr(message_r, "headers", None)
                            or getattr(item_r, "response_headers", None)
                            or getattr(item_r, "headers", None)
                            or {}
                        )
                        if isinstance(headers_r, dict):
                            cache_status_r = headers_r.get("anthropic-cache-status") or headers_r.get("x-anthropic-cache-status")
                            cache_token_credits_r = headers_r.get("anthropic-cache-token-credits")
                        # Also check usage object
                        if isinstance(usage_r, dict):
                            cache_status_r = cache_status_r or usage_r.get("anthropic-cache-status")
                            cache_token_credits_r = cache_token_credits_r or usage_r.get("anthropic-cache-token-credits")
                        if cache_status_r or cache_token_credits_r:
                            logger.info(
                                "[batch][resume][cache] custom_id=%s status=%s token_credits=%s", custom_id_r, cache_status_r, cache_token_credits_r
                            )
                        else:
                            logger.info("[batch][resume][cache] custom_id=%s status=%s", custom_id_r, None)
                    except Exception:
                        logger.debug("Failed to read cache headers for resume custom_id=%s", custom_id_r)
                    
                    if stop_reason_r == "pause_turn":
                        next_round.append({
                            "custom_id": custom_id_r,
                            "row": prior["row"],
                            "original_user_text": prior.get("original_user_text", ""),
                            "combined_blocks": combined_so_far,
                        })
                        continue
                    
                    # Terminal: finalize
                    r_row = prior["row"]
                    filename_r = create_output_filename(r_row)
                    filepath_r = os.path.join(output_dir, filename_r)
                    resumed_parts = [b.get("text", "") for b in combined_so_far if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
                    text_final = ("\n".join(resumed_parts)).strip() if resumed_parts else ""
                    raw_for_parse_r = text_final
                    # Try to parse JSON; on failure, fall back to writing raw text to keep output file always written
                    try:
                        obj_r = json.loads(raw_for_parse_r)
                    except Exception as e_json:
                        logger.warning("[resume] JSON parse failed for %s: %s. Will save RAW text.", custom_id_r, e_json)
                        obj_r = None
                    
                    # Write output (parsed JSON if available, else raw text)
                    try:
                        with open(filepath_r, "w", encoding="utf-8") as out:
                            if obj_r is not None:
                                json.dump(obj_r, out, ensure_ascii=False, indent=2)
                            else:
                                out.write(raw_for_parse_r)
                    except Exception as e_write:
                        logger.error("[resume] Failed to write output for %s: %s", filename_r, e_write)
                        update_checkpoint(output_dir, r_row, False, load_checkpoint(output_dir))
                        fail += 1
                        continue
                    ing_ok_r = True
                    if not no_ingest:
                        ing_ok_r = trigger_ingestion(filepath_r, logger)
                    else:
                        logger.info("[resume] Auto-ingest disabled; skipping ingestion for %s", filepath_r)
                    update_checkpoint(output_dir, r_row, ing_ok_r, load_checkpoint(output_dir))
                    if ing_ok_r:
                        ok += 1
                    else:
                        fail += 1
                
                # Prepare for another follow-up round if any still paused
                resume_queue = next_round
            
        except Exception as e:
            logger.error("[resume] Error processing batch %s: %s", batch_id, e)
            # Mark all rows in this batch as failed
            for row_minimal in rows_minimal:
                full_row = {
                    "brand": row_minimal.get("brand", ""),
                    "product_name": row_minimal.get("product_name", ""),
                    "shade_of_lipstick": row_minimal.get("shade_of_lipstick", ""),
                    "sku": row_minimal.get("sku", "")
                }
                update_checkpoint(output_dir, full_row, False, load_checkpoint(output_dir))
            fail += len(rows_minimal)
    
    return ok, fail, total_input_tokens, total_output_tokens


def build_user_message(product_row: Dict[str, str], use_natural_generation: bool = True) -> Tuple[str, str]:
    # Map CSV fields to the schema keys commonly used in your lipstick template
    # Validate required fields before constructing the payload
    required_keys = ["brand", "product_name", "shade_of_lipstick", "s_no"]
    missing = [k for k in required_keys if not (product_row.get(k) or "").strip()]
    if missing:
        raise ValueError(f"Missing required fields for user message: {', '.join(missing)}")

    product_for_model = {
        "brand": product_row["brand"],
        "product_line": product_row["product_name"],  # maps to template's product_line
        "shade": product_row["shade_of_lipstick"],
        "full_name": product_row["full_name"],
        "sku": product_row.get("sku", ""),
        "category": product_row.get("category", ""),
        "sub_category": product_row.get("sub_category", ""),
        "leaf_level_category": product_row.get("leaf_level_category", ""),
        "s_no": product_row.get("s_no", ""),
    }

    if use_natural_generation:
        # Dynamic per-product message (non-cacheable). System is supplied elsewhere for batching.
        user_text = textwrap.dedent(
            f"""
            Product Information:
            {json.dumps(product_for_model, ensure_ascii=False, indent=2)}
            """
        ).strip()
        return "", user_text
    else:
        raise ValueError("use_natural_generation=False is not supported. Refusing to build an alternate prompt shape.")



# ------------------------------
# Anthropic call w/ built-in web search tool
# ------------------------------

def call_claude_with_web_search(api_key: str, model: str, user_content: str, max_tokens: int = 32000, debug_basepath: str | None = None, system_text: str | None = None, use_cache: bool = True, request_timeout: tuple[int, int] = (30, 1800)) -> tuple[str, dict]:
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
                "max_uses": 10,
            }
        ],
    }

    # Mirror earlier behavior: include thinking block when enabled
    if ENABLE_THINKING:
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": 31999,
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
        resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
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



# ------------------------------
# OpenAI GPT-5 call w/ web_search and thinking
# ------------------------------

def call_gpt_with_web_search(
    api_key: str,
    model: str,
    user_content: str,
    max_tokens: int = 128000,
    debug_basepath: str | None = None,
    enable_thinking: bool = True,
    enable_web_search: bool = True,
    system_text: str | None = None,
) -> tuple[str, dict]:
    """Call OpenAI Responses API with GPT-5 using web_search tool and optional thinking.
    Returns (raw_text, usage_info_dict).
    """
    logger = logging.getLogger(__name__)
    client = OpenAI(api_key=api_key)

    # Build messages array so we can include the prompt as a system role
    messages: List[Dict[str, Any]] = []
    if system_text and system_text.strip():
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_content})

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_tokens,
    }
    if enable_thinking:
        kwargs["reasoning"] = {"effort": "high"}
    # Encourage detailed text output
    kwargs["text"] = {"verbosity": "high"}
    if enable_web_search:
        kwargs["tools"] = [{"type": "web_search"}]
    kwargs["max_output_tokens"] = max_tokens

    t0 = time.time()
    resp = client.responses.create(**kwargs)
    dt = time.time() - t0

    # Extract output text across SDK variants
    output_text = None
    try:
        output_text = getattr(resp, "output_text", None)
        if not output_text:
            # Try assembling from outputs list
            parts = []
            for item in getattr(resp, "output", []) or []:
                txt = getattr(item, "content", None) or getattr(item, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
            if parts:
                output_text = "\n".join(parts)
    except Exception:
        output_text = None

    raw_text = (output_text or "").strip()

    # Usage best-effort extraction
    usage_info: Dict[str, Any] = {}
    try:
        usage = getattr(resp, "usage", None)
        if isinstance(usage, dict):
            usage_info["input_tokens"] = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            usage_info["output_tokens"] = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        else:
            usage_info["input_tokens"] = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
            usage_info["output_tokens"] = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    except Exception:
        usage_info.setdefault("input_tokens", 0)
        usage_info.setdefault("output_tokens", 0)

    # Optional debug sidecars
    if debug_basepath:
        try:
            with open(f"{debug_basepath}_response_text.txt", "w", encoding="utf-8") as f_raw:
                f_raw.write(raw_text)
            with open(f"{debug_basepath}_response_usage.json", "w", encoding="utf-8") as f_usage:
                json.dump(usage_info, f_usage, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("Failed writing GPT debug sidecars for %s", debug_basepath)

    try:
        logger.info("GPT Response <- time=%.2fs usage=%s", dt, usage_info)
    except Exception:
        pass

    return raw_text, usage_info


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
    parser.add_argument("--no_cache", action="store_true", help="Disable Anthropic prompt caching (use for token price comparison)")
    parser.add_argument("--resume_batches", action="store_true", help="Resume processing of pending batches that timed out during polling")
    args, unknown = parser.parse_known_args()
    # Log received CLI for diagnostics
    try:
        logging.getLogger(__name__).info("CLI argv: %s", " ".join(sys.argv))
        logging.getLogger(__name__).info("Parsed flags: no_batch=%s, no_ingest=%s, no_cache=%s, resume_batches=%s", args.no_batch, args.no_ingest, args.no_cache, args.resume_batches)
    except Exception:
        pass
    # Hardcoded file paths
    prompt_path = "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/data/beauty_prompt_restored.json"
    
    # Data source selection based on USE_EXCEL_DATA parameter
    if USE_EXCEL_DATA:
        # input_data_path = "https://kult20256-my.sharepoint.com/:x:/g/personal/harshit_a_kult_app/ER4mU46_r9VAu0XCFYkVzD8Be1E4BFyHPulmXQYVf0ZTtQ?e=H0FrNA"
        input_data_path = "https://kult20256-my.sharepoint.com/:x:/g/personal/srushti_kult_app/EX5Vg0krs25Ivpvupcn4_uIBTOmgOmadWw82laanvdeh2g?e=fGfOBn"
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

    # Select provider and validate API keys
    provider = None
    if USE_GPT and USE_CLAUDE:
        provider = "claude"  # Prefer Claude when both are enabled
        logger.warning("Both USE_GPT and USE_CLAUDE are True; defaulting to Claude. Set USE_CLAUDE=False to use GPT.")
    elif USE_GPT:
        provider = "gpt"
    elif USE_CLAUDE:
        provider = "claude"
    else:
        logger.error("No provider enabled. Set USE_GPT=True or USE_CLAUDE=True.")
        sys.exit(2)

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if provider == "claude":
        if not anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY is not set")
            sys.exit(2)
    else:
        if not openai_api_key:
            logger.error("OPENAI_API_KEY is not set for GPT provider")
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

    # RESUME MODE: If --resume_batches is passed, process pending batches and exit
    if args.resume_batches:
        if provider != "claude":
            logger.info("Resume batches is only applicable to Claude batch runs. GPT provider selected; nothing to resume.")
            sys.exit(0)
        logger.info("🔄 RESUME MODE: Processing pending batches...")
        logger.info("="*60)
        
        # Load prompt (needed for resume processing)
        try:
            prompt_text = read_prompt(prompt_path)
            logger.info(f"Loaded prompt from: {prompt_path}")
        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            sys.exit(1)
        
        # Call resume function
        resume_start = time.time()
        ok, fail, in_tok, out_tok = resume_pending_batches(
            api_key=anthropic_api_key,
            model=model,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            output_dir=output_dir,
            logger=logger,
            no_ingest=args.no_ingest,
            use_cache=use_prompt_cache,
        )
        resume_time = time.time() - resume_start
        
        # Print summary and exit
        logger.info("="*60)
        logger.info("Batch Resume Completed")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration: {resume_time:.2f} seconds ({resume_time/60:.2f} minutes)")
        logger.info("")
        logger.info("📊 RESUME STATISTICS:")
        logger.info(f"   Products processed: {ok + fail}")
        logger.info(f"   Successful: {ok}")
        logger.info(f"   Failed: {fail}")
        logger.info(f"   Success rate: {(ok/(ok+fail)*100):.1f}%" if (ok+fail) > 0 else "N/A")
        logger.info("")
        logger.info(f"💰 Token usage: {in_tok:,} input + {out_tok:,} output = {in_tok + out_tok:,}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info("="*60)
        sys.exit(0)

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
    if provider == "claude" and not args.no_batch:
        batch_size = BATCH_SIZE_DEFAULT
        logger.info("Running in BATCH mode (default): batch_size=%s", batch_size)
        ok, fail, batch_in_tokens, batch_out_tokens = run_batch_generation(
            api_key=anthropic_api_key,
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
        
        # Compute per-product debug base path early 
        debug_basepath = None
        if debug_root:
            base_no_ext = os.path.splitext(filename)[0]
            debug_basepath = os.path.join(debug_root, base_no_ext)

        # Build user message for this row. We intentionally ignore the returned system_text (empty)
        # and pass prompt_text as the system message so sync path mirrors batch behavior.
        _, user_msg = build_user_message(row, use_natural_generation=True)
        
        if provider == "claude":
            try:
                raw, usage_info = call_claude_with_web_search(
                    anthropic_api_key,
                    model,
                    user_msg,
                    max_tokens=max_tokens,
                    debug_basepath=debug_basepath,
                    system_text=prompt_text,
                    use_cache=use_prompt_cache,
                    request_timeout=(30, 1800),
                )

                # Pre-clean: many providers wrap JSON in ```json fences; extract first JSON block if present
                raw_for_parse = raw

                try:
                    obj = json.loads(raw_for_parse)
                except Exception:
                    # If parsing fails for any reason, save RAW to main file (batch-parity behavior)
                    obj = None

                # Save main product file (parsed JSON if available, else raw text)
                with open(filepath, "w", encoding="utf-8") as out:
                    if obj is not None:
                        json.dump(obj, out, ensure_ascii=False, indent=2)
                    else:
                        out.write(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, indent=2))

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
        elif provider == "gpt":
            try:
                raw, usage_info = call_gpt_with_web_search(
                    openai_api_key,
                    "gpt-5",
                    user_msg,
                    max_tokens=128000,
                    debug_basepath=debug_basepath,
                    enable_thinking=ENABLE_THINKING,
                    enable_web_search=True,
                    system_text=prompt_text,
                )

                # Pre-clean: many providers wrap JSON in ```json fences; extract first JSON block if present
                raw_for_parse = raw

                try:
                    obj = json.loads(raw_for_parse)
                except Exception as e:
                    # Attempt to extract JSON object if assistant added prose before/after
                    cleaned = raw
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
                
                # Track usage
                input_tokens = usage_info.get("input_tokens", 0)
                output_tokens = usage_info.get("output_tokens", 0)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Save file
                with open(filepath, "w", encoding="utf-8") as out:
                    try:
                        # Try to parse and save clean JSON
                        json.dump(obj, out, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # If parsing fails, write raw text as fallback
                        out.write(raw)

                # Always write a raw sidecar like batch mode (<base>.raw.result.txt)
                try:
                    base_no_ext = os.path.splitext(filename)[0]
                    raw_result_txt_path = os.path.join(output_dir, f"{base_no_ext}.raw.result.txt")
                    with open(raw_result_txt_path, "w", encoding="utf-8") as f_rawtxt:
                        f_rawtxt.write(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, indent=2))
                except Exception as _e:
                    logger.debug("Failed to write raw result text sidecar for %s: %s", filename, _e)

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
                except Exception as _:
                    logger.debug("Failed to save raw sidecar file: %s", _)
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