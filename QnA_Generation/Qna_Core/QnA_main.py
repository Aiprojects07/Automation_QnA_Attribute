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

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-4-1-20250805"  # From your example

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
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
        # Check if it's a JSON file with a "prompt" key
        if path.endswith('.json'):
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "prompt" in data:
                    return data["prompt"]
            except json.JSONDecodeError:
                pass
        
        # Return raw content if not JSON or no "prompt" key
        return content


def read_schema_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    brand = m.get("brand") or m.get("company")
    product_name = (
        m.get("product_name")
        or m.get("product")
        or m.get("product_title")
        or m.get("line")
        or m.get("product_line")
    )
    shade = m.get("shade_of_lipstick") or m.get("shade") or m.get("color")
    # Support multiple SKU header variants including new 'Kult SKU Code'
    sku = (
        m.get("sku")
        or m.get("sku_code")
        or m.get("product_sku")
        or m.get("item_code")
        or m.get("kult_sku_code")
        or ""
    )
    # Optional category support
    category = m.get("category") or m.get("product_category") or m.get("category_name") or ""
    # Optional sub-category (handle hyphenated and concatenated variants)
    sub_category = (
        m.get("sub_category")
        or m.get("subcategory")
        or m.get("sub-category")
        or ""
    )
    # Optional leaf level category
    leaf_level_category = (
        m.get("leaf_level_category")
        or m.get("leaflevelcategory")
        or m.get("leaf_level_cat")
        or m.get("sub_sub_category")  # new column name maps to leaf-level category
        or ""
    )
    # Optional color/appearance metrics (support multiple header variants)
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
    return {
        "brand": brand,
        "product_name": product_name,
        "shade_of_lipstick": shade,
        "sku": sku,
        "category": category,
        "sub_category": sub_category,
        "leaf_level_category": leaf_level_category,
        # New optional fields
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
    }


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: List[Dict[str, str]] = []
        for i, r in enumerate(reader, 1):
            try:
                out.append(normalize_row(r))
            except Exception as e:
                raise ValueError(f"Row {i} invalid: {e}") from e
        if not out:
            raise ValueError("Input CSV contained no rows")
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


def build_user_message(prompt_text: str, product_row: Dict[str, str], schema_obj: Dict[str, Any] = None, use_natural_generation: bool = True) -> Tuple[str, str]:
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
    }

    if use_natural_generation:
        # Build a STATIC system message (cacheable) with prompt_text and REQUIRED JSON FORMAT (no product interpolation)
        if schema_obj:
            required_format = json.dumps(schema_obj, ensure_ascii=False, indent=2)
        else:
            # Minimal static template if schema is not provided
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
                          "solution": "solution here (include this key even if empty)"
                        }
                      ]
                    }
                  ]
                }
                """
            ).strip()

        system_text = textwrap.dedent(
            f"""
            {prompt_text}

            CRITICAL FORMATTING REQUIREMENTS:
            - Return ONLY valid JSON (no markdown, no commentary)
            - Treat the JSON below as a SHAPE EXAMPLE (keys and nesting) only; do NOT copy the number of sections or QAs from it
            - Follow the section and QA counts from the PROMPT'S STRUCTURE (exactly 5 sections, with the required QAs per section)
            - Expand arrays to meet the PROMPT requirements even if the example shows fewer items
            - Use the exact product keys shown (including "sku", "category", "sub_category", "leaf_level_category"); keys must always be present (empty string allowed if unknown)
            - Do NOT include any citations, footnotes, source markers, or attribution (e.g., <cite ...>...</cite>, [1], (ref), URLs). Present everything as expert knowledge.

            REQUIRED JSON FORMAT:
            {required_format}
            
            Additional requirements:
            - Under product, include keys: "category", "sub_category", and "leaf_level_category".
            - These keys must always be present (empty string allowed if not provided).
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
        # Use fixed template (fallback): system contains prompt + schema; user contains product info
        system_text = textwrap.dedent(
            f"""
            <<PROMPT>>
            {prompt_text}

            <<OUTPUT FORMAT>>
            You MUST output valid JSON only that exactly matches this format (keys and types):
            {json.dumps(schema_obj or {}, ensure_ascii=False, indent=2)}
            """
        ).strip()
        user_text = textwrap.dedent(
            f"""
            <<PRODUCT>>
            {json.dumps(product_for_model, ensure_ascii=False, indent=2)}
            """
        ).strip()
        return system_text, user_text


# ------------------------------
# Anthropic call w/ built-in web search tool
# ------------------------------

def call_claude_with_web_search(api_key: str, model: str, user_content: str, max_tokens: int = 30000, debug_basepath: str | None = None, system_text: str | None = None) -> tuple[str, dict]:
    """Call Claude API with retry logic and usage tracking."""
    logger = logging.getLogger(__name__)
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        # Enable prompt caching (beta). This allows cache_control on the system block to be used and
        # returns cache-related headers we can log.
        "anthropic-beta": "prompt-caching-2024-07-31",
    }
    # Headers prepared
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "messages": [
            {"role": "user", "content": user_content}
        ],
        # IMPORTANT: Use Anthropic's built-in web search tool per your example
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 4
            }
        ],
    }
    
    # Inject cacheable system message (prompt caching) if provided
    if system_text:
        payload["system"] = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    
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
        
        # The response content is a list of blocks; accumulate only text parts
        parts = []
        for c in data.get("content", []):
            ctype = c.get("type")
            if ctype == "text":
                parts.append(c.get("text", ""))
            else:
                # Tool activity logging disabled per user request; ignore non-text blocks
                continue

        text = "".join(parts).strip()

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
# Main
# ------------------------------

def main() -> None:
    # CLI args (only debug for now)
    parser = argparse.ArgumentParser(description="Lipstick QnA generator with web search")
    parser.add_argument("--debug", action="store_true", help="Enable per-product debug dumps under output/debug/")
    parser.add_argument("--no_ingest", action="store_true", help="Disable automatic Pinecone ingestion after writing each JSON output")
    args, unknown = parser.parse_known_args()
    # Hardcoded file paths
    prompt_path = "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/data/lipstick-qa-prompt-builder.json"
    input_csv = "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/data/lipstick_list.csv"
    output_dir = "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/output"
    model = DEFAULT_MODEL
    max_tokens = 30000

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

    # Load configuration files
    logger.info("Loading configuration files...")
    try:
        prompt_text = read_prompt(prompt_path)
        logger.info(f"Loaded prompt from: {prompt_path}")
        
        # Use inline shape-only REQUIRED JSON FORMAT; no external schema file
        schema_obj = None
        logger.info("Using inline shape-only REQUIRED JSON FORMAT (no external schema file)")
        
        rows = read_rows(input_csv)
        logger.info(f"Loaded {len(rows)} products from: {input_csv}")
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

    # One-time Helicone status log flag
    helicone_status_logged = False

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
                update_checkpoint(output_dir, row, ing_ok, checkpoint_data)
                if not ing_ok:
                    logger.warning("Marked as failed due to ingestion failure for existing file: %s", filename)
            else:
                update_checkpoint(output_dir, row, True, checkpoint_data)
                logger.info("Auto-ingest disabled via --no_ingest; marking existing file as completed: %s", filename)
            continue
        
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

                        # Optional debug: save parsed JSON as well
                        if debug_basepath:
                            try:
                                with open(f"{debug_basepath}_parsed.json", "w", encoding="utf-8") as f_parsed:
                                    json.dump(obj, f_parsed, ensure_ascii=False, indent=2)
                            except Exception:
                                logger.debug("Failed to write parsed JSON for %s", debug_basepath)

                        # Trigger ingestion unless disabled
                        ing_ok = True
                        if not args.no_ingest:
                            ing_ok = trigger_ingestion(filepath, logger)
                        else:
                            logger.info("Auto-ingest disabled via --no_ingest; skipping ingestion for %s", filepath)

                        # Update checkpoint and continue to next product
                        update_checkpoint(output_dir, row, ing_ok, checkpoint_data)
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

        system_text, user_msg = build_user_message(prompt_text, row, schema_obj, use_natural_generation=True)

        # Compute per-product debug base path
        debug_basepath = None
        if debug_root:
            base_no_ext = os.path.splitext(filename)[0]
            debug_basepath = os.path.join(debug_root, base_no_ext)
        
        try:
            raw, usage_info = call_claude_with_web_search(
                api_key,
                model,
                user_msg,
                max_tokens=max_tokens,
                debug_basepath=debug_basepath,
                system_text=system_text,
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

            # Write raw-response and audit file to measure dropped content (if any)
            try:
                base_no_ext = os.path.splitext(filename)[0]
                # Full raw text from the LLM
                raw_full_txt_path = os.path.join(output_dir, f"{base_no_ext}.raw.full.txt")
                with open(raw_full_txt_path, "w", encoding="utf-8") as f_raw_full:
                    f_raw_full.write(raw)

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
                logger.debug("Failed to write raw/audit sidecar files for %s: %s", filename, _e)

            # Optional debug: save parsed JSON as well (duplicate of output for quick diff)
            if debug_basepath:
                try:
                    with open(f"{debug_basepath}_parsed.json", "w", encoding="utf-8") as f_parsed:
                        json.dump(obj, f_parsed, ensure_ascii=False, indent=2)
                except Exception:
                    logger.debug("Failed to write parsed JSON for %s", debug_basepath)

            # Trigger Pinecone ingestion for this freshly written JSON file unless disabled
            ing_ok = True
            if not args.no_ingest:
                ing_ok = trigger_ingestion(filepath, logger)
            else:
                logger.info("Auto-ingest disabled via --no_ingest; skipping ingestion for %s", filepath)
             
            product_time = time.time() - product_start_time
            session_ok += 1
            
            # Update checkpoint
            update_checkpoint(output_dir, row, ing_ok, checkpoint_data)
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
            update_checkpoint(output_dir, row, False, checkpoint_data)
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