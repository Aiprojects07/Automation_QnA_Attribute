import os
import json
import csv
import sys
import logging
from datetime import datetime
from io import BytesIO
from pinecone import Pinecone, ServerlessSpec
import anthropic  # Claude API
try:
    # Optional typed helpers (available in newer SDKs)
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as BatchRequest
except Exception:
    MessageCreateParamsNonStreaming = None  # type: ignore
    BatchRequest = None  # type: ignore
from dotenv import load_dotenv
import pandas as pd
import requests
from openpyxl import load_workbook
from openai import OpenAI
import textwrap
from typing import Any, Dict, List, Tuple

# =============================
# 🔧 CONFIGURATION
# =============================

# Load environment variables from .env (if present)
load_dotenv()

# Keys and index come from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Prompt file (JSON) containing a "prompt" key
PROMPT_PATH = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/data/attribute-extraction-prompt-json (1).json"
# Data source configuration
USE_EXCEL_DATA = True  # Set to False to use CSV
DATA_SOURCE_PATH = "https://kult20256-my.sharepoint.com/:x:/g/personal/harshit_a_kult_app/ER4mU46_r9VAu0XCFYkVzD8Be1E4BFyHPulmXQYVf0ZTtQ?rtime=Yly5qN0K3kg"

BATCH_SIZE = 1
REGISTRY_PATH = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/output"
OUTPUT_DIR = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/maximum_and_common_Attributes_per_product"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: automatically upsert generated JSONs to Pinecone after batch finishes
# Set to True to enable auto-upsert via common_and_max_attributes_upserting.py
RUN_COMMON_ATTR_UPSERT = False

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in environment/.env")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set in environment/.env")
if not INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME is not set in environment/.env")

# =============================
# 🔌 INITIALIZE CLIENTS
# =============================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# OpenAI client for embeddings
oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# =============================
# 📋 LOGGING SETUP
# =============================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration with both file and console handlers."""
    # Create logs directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"attribute_generation_{timestamp}.log")
    
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

# =============================
# 📄 Prompt loader
# =============================

def read_prompt(path: str) -> str:
    """Read the prompt text. Supports JSON with a top-level "prompt" field or raw text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    # Log which prompt file is being used
    try:
        logging.getLogger(__name__).info("\ud83d\udcc4 Using prompt file: %s", path)
    except Exception:
        pass
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
    # Try JSON with {"prompt": "..."}
    try:
        obj = json.loads(data)
        if isinstance(obj, dict) and "prompt" in obj:
            return str(obj["prompt"]).strip()
    except Exception:
        pass
    # Fallback: raw content
    return data

ATTRIBUTE_PROMPT_TEMPLATE = None  # Will be loaded at runtime from PROMPT_PATH

# =============================
# 📊 DATA SOURCE HELPERS
# =============================

def _build_shared_system_blocks(prompt_text: str, use_cache: bool) -> List[Dict[str, Any]]:
    """Return a system array with a small uncached header + a large cached block.
    Keeps the large block byte-stable and marks it with cache_control so batches can reuse it.
    Always uses inline REQUIRED JSON FORMAT (no external schema).
    """
    # Small uncached header (stable and small)
    header = {
        "type": "text",
        "text": textwrap.dedent(
            """
            Follow the formatting and constraints exactly.
            Output strictly valid JSON only. No markdown.
            """
        ).strip() + "\n",
    }

    big_block = {
        "type": "text",
        "text": textwrap.dedent(
            f"""
            {prompt_text}

            CRITICAL FORMATTING REQUIREMENTS:
            - Do NOT include code fences (```), markdown, prose, or any text before/after the JSON
            - Start your response with '[' or '{{' and return exactly one complete JSON value
            """
        ).strip(),
    }

    # DIAGNOSTIC: log what use_cache actually is here
    try:
        logging.getLogger(__name__).info(
            "_build_shared_system_blocks(attr): use_cache=%r (type=%s)", use_cache, type(use_cache).__name__
        )
    except Exception:
        pass

    if use_cache:
        big_block["cache_control"] = {"type": "ephemeral"}

    return [header, big_block]

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


def normalize_row(raw: dict) -> dict:
    """Normalize row data from Excel or CSV to a consistent format."""
    # Normalize headers and values, strip periods to handle headers like 'S.No.'
    m = {(k or "").strip().lower().replace(" ", "_").replace(",", "").replace(".", ""): (v or "").strip() for k, v in raw.items()}
    
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
    
    # Sub-category mapping
    sub_category = (
        m.get("sub_category")
        or m.get("subcategory")
        or m.get("sub-category")
        or ""
    )
    
    # Leaf level category
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
        or m.get("serial_number")
        or m.get("serial_no")
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


def read_rows(data_path: str) -> list:
    """Read product data from either Excel (SharePoint URL) or CSV file based on USE_EXCEL_DATA setting."""
    logger = logging.getLogger(__name__)
    out = []
    
    if USE_EXCEL_DATA:
        # Handle Excel data from SharePoint URL
        try:
            # Convert SharePoint share URL to direct download URL
            download_url = convert_sharepoint_url(data_path)
            logger.info(f"[excel] original_url={data_path}")
            logger.info(f"[excel] converted_download_url={download_url}")
            
            # Download and read Excel file
            headers = {"User-Agent": "Mozilla/5.0 (compatible; Attribute-Script/1.0)"}
            response = requests.get(download_url, headers=headers, timeout=30)
            if not response.ok:
                logger.error(f"[excel] GET failed: status={response.status_code} url={response.url}")
                # Fallback: some tenants use guestaccess.aspx instead of download.aspx
                if response.status_code == 404 and 'download.aspx' in download_url:
                    alt_url = download_url.replace('download.aspx', 'guestaccess.aspx')
                    logger.info(f"[excel] retrying with guestaccess.aspx: {alt_url}")
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

# =============================
# ⚙️ MAIN PROCESSING LOGIC
# =============================

def embed_query(text: str) -> list:
    """Convert text into an embedding vector using OpenAI."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    
    resp = oai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


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
    sku: str = None,
) -> dict:
    """Build a Pinecone-compatible metadata filter.
    
    Pass strings for exact match or lists for $in matching.
    """
    filt = {}
    for key, val in [
        ("sku", sku),
    ]:
        norm = _normalize_filter_value(val)
        if norm is not None:
            filt[key] = norm
    
    return filt


def fetch_chunks_for_sku(sku: str, product_info: dict = None):
    """Retrieve all chunks from Pinecone for a given SKU using semantic search.
    
    Args:
        sku: Product SKU to search for
        product_info: Dict containing category, sub_category, leaf_level_category for filtering
    
    Returns:
        List of text chunks from Pinecone
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Use SKU as the search query
        query_text = sku
        
        # Build metadata filter using helper function (only SKU)
        metadata_filter = build_metadata_filter(sku=sku)
        
        logger.info(f"Querying Pinecone for SKU: {sku}, filter: {metadata_filter}")
        
        # Embed the SKU for semantic search
        query_vector = embed_query(query_text)
        
        # Query Pinecone with semantic search + metadata filters
        results = index.query(
            vector=query_vector,
            filter=metadata_filter,
            top_k=100,  # Get all relevant chunks
            namespace=PINECONE_NAMESPACE,  # IMPORTANT: Must specify namespace
            include_metadata=True
        )
        
        matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])
        logger.info(f"Found {len(matches)} matches for SKU: {sku}")
        
        chunks = []
        for m in matches:
            md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
            # Try multiple possible content field names
            text = md.get("content") or md.get("text") or md.get("chunk_text") or ""
            if text:
                chunks.append(text)
                logger.debug(f"Added chunk with {len(text)} characters for SKU: {sku}")
        
        if not chunks:
            logger.warning(f"No text content found in metadata for SKU: {sku}")
            if matches:
                # Log the first match's metadata keys to help debug
                sample_metadata = matches[0].get("metadata", {}) if isinstance(matches[0], dict) else getattr(matches[0], "metadata", {})
                logger.warning(f"Sample metadata keys: {list(sample_metadata.keys())}")
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error fetching chunks for SKU {sku}: {e}")
        return []


def extract_attributes_with_claude(content: str, *, use_cache: bool = True) -> str:
    """Send combined product text to Claude and get attribute list (as JSON text).
    Uses prompt caching for the large, stable system content when use_cache=True.
    """
    global ATTRIBUTE_PROMPT_TEMPLATE
    if ATTRIBUTE_PROMPT_TEMPLATE is None:
        ATTRIBUTE_PROMPT_TEMPLATE = read_prompt(PROMPT_PATH)

    # Build system content (header + big cached block)
    system_blocks = _build_shared_system_blocks(ATTRIBUTE_PROMPT_TEMPLATE, use_cache)

    # User message carries the product-specific content only (kept small and uncached)
    user_content = [
        {
            "type": "text",
            "text": (
                "Product Content:\n" 
                "----------------\n" 
                f"{content}\n" 
                "----------------\n"
                "Output only JSON, no explanations."
            ),
        }
    ]

    # Stream response
    response_text = ""
    with client.messages.stream(
        model="claude-opus-4-1-20250805",
        max_tokens=30000,
        system=system_blocks,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.2,
    ) as stream:
        for text in stream.text_stream:
            response_text += text

    return response_text.strip()


def _make_custom_id(row: Dict[str, Any]) -> str:
    sku = (row.get("sku") or "").strip()
    if sku:
        return sku
    # Fallback: compose from brand/product/shade
    return f"{(row.get('brand') or '').strip()}-{(row.get('product_name') or '').strip()}-{(row.get('shade_of_lipstick') or '').strip()}" or "unknown"


def run_batch_attributes(
    rows: List[Dict[str, Any]],
    *,
    model: str = "claude-opus-4-1-20250805",
    max_tokens: int = 8192,
    batch_size: int = 2,
    use_cache: bool = True,
    output_dir: str,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """Submit products to Anthropic Messages Batch API in waves and write JSON outputs.
    Returns (ok_count, fail_count).
    """
    global ATTRIBUTE_PROMPT_TEMPLATE
    if ATTRIBUTE_PROMPT_TEMPLATE is None:
        ATTRIBUTE_PROMPT_TEMPLATE = read_prompt(PROMPT_PATH)

    # Build shared system blocks once (cacheable)
    shared_system = _build_shared_system_blocks(ATTRIBUTE_PROMPT_TEMPLATE, use_cache)
    try:
        logger.info("Prompt caching (batch): %s", "ENABLED" if use_cache else "DISABLED")
    except Exception:
        pass

    ok, fail = 0, 0

    for i in range(0, len(rows), batch_size):
        wave = rows[i:i + batch_size]
        requests_payload: List[Any] = []

        logger.info(f"[batch][attr] Processing wave {i//batch_size + 1}: {len(wave)} items")

        for row in wave:
            # In our attribute pipeline, we fetch chunks first; combine Pinecone chunks on the fly for each row
            sku = (row.get("sku") or "").strip()
            chunks = fetch_chunks_for_sku(sku)
            combined_text = "\n\n".join(chunks) if chunks else ""

            # Build per-product user content (kept small and uncached)
            # We pass only the combined_text; all formatting rules live in the system prompt.
            content = [
                {
                    "type": "text",
                    "text": f"{combined_text}",
                }
            ]

            params: Dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "system": shared_system,
                "messages": [{"role": "user", "content": content}],
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
                requests_payload.append({"custom_id": custom_id, "params": params})

        # Submit batch
        logger.info("[batch][attr] Submitting wave %s: %s requests", i//batch_size + 1, len(requests_payload))
        batch = client.messages.batches.create(requests=requests_payload)
        batch_id = getattr(batch, "id", None) or batch["id"]  # type: ignore
        logger.info("[batch][attr] Batch submitted: id=%s", batch_id)

        # Poll until completion
        import time as _time
        start_poll = _time.time()
        max_wait_seconds = 45 * 60  # 30 minutes
        while True:
            status_obj = client.messages.batches.retrieve(batch_id)
            status = getattr(status_obj, "processing_status", None)
            logger.info("[batch][attr] Batch %s status: %s", batch_id, status)
            if status in ("ended", "completed", "cancelled", "expired", "failed"):
                break
            if (_time.time() - start_poll) >= max_wait_seconds:
                logger.warning("[batch][attr] Batch %s polling timed out; moving to next wave.", batch_id)
                break
            _time.sleep(3)

        # Retrieve results
        results_iter = client.messages.batches.results(batch_id)
        try:
            items = list(results_iter)
        except Exception:
            items = getattr(results_iter, "data", None) or results_iter.get("data", [])  # type: ignore

        logger.info("[batch][attr] Batch %s returned %s results", batch_id, len(items))

        # Process results
        for item in items:
            custom_id = getattr(item, "custom_id", None) or item.get("custom_id")  # type: ignore
            result_obj = getattr(item, "result", None) or item.get("result")  # type: ignore
            if getattr(result_obj, "type", None) != "succeeded":
                # Try to log error
                err_wrapper = getattr(result_obj, "error", None)
                err_inner = getattr(err_wrapper, "error", None)
                err_msg = (
                    getattr(err_inner, "message", None)
                    or getattr(err_wrapper, "message", None)
                    or None
                )
                if err_msg:
                    logger.error("[batch][attr] Item failed: %s -> %s", custom_id, err_msg)
                fail += 1
                continue

            message = getattr(result_obj, "message", None)
            # Locate row by custom_id (sku)
            row = next((r for r in wave if _make_custom_id(r) == custom_id), None)
            if not row:
                logger.warning("[batch][attr] Result custom_id=%s has no matching row", custom_id)
                fail += 1
                continue

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
                        "[batch][attr][cache] custom_id=%s status=%s token_credits=%s", custom_id, cache_status, cache_token_credits
                    )
                else:
                    logger.info("[batch][attr][cache] custom_id=%s status=%s", custom_id, None)
            except Exception:
                logger.debug("[batch][attr] Failed to read cache headers for custom_id=%s", custom_id)

            # Usage metrics (input/output tokens and cache reads)
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
                logger.info("[batch][attr][usage] custom_id=%s in=%s out=%s cache_read_in=%s", custom_id, in_tok, out_tok, cache_read_tok)
            except Exception:
                pass

            # Extract text parts
            try:
                content_blocks = getattr(message, "content", None) or []
                parts = []
                for c in content_blocks:
                    if getattr(c, "type", None) == "text":
                        parts.append(getattr(c, "text", ""))
                text_payload = "".join(parts)
            except Exception:
                text_payload = ""

            # Save output as <sku>.json
            sku = (row.get("sku") or "").strip() or custom_id or "unknown"
            output_file = os.path.join(output_dir, f"{sku}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text_payload)

            # Update row status so caller can update registry
            row["status"] = "done"
            row["attributes_file"] = output_file
            row["timestamp"] = datetime.utcnow().isoformat()
            ok += 1

    return ok, fail


def process_batch(batch):
    """Process a batch of products, fetching all chunks per SKU and asking Claude for attributes."""
    logger = logging.getLogger(__name__)
    for product in batch:
        sku = product.get("sku", "").strip()
        if not sku:
            logger.warning("⚠️ Missing SKU in product; skipping")
            product["status"] = "error"
            continue

        logger.info(f"🔍 Processing {sku} - {product.get('product_name', '')}")

        # Fetch chunks using only SKU filter
        chunks = fetch_chunks_for_sku(sku)
        if not chunks:
            logger.warning(f"⚠️ No chunks found for {sku}")
            product["status"] = "missing"
            continue

        # Combine product chunks to maximize attribute coverage
        combined_text = "\n\n".join(chunks)
        attributes_json = extract_attributes_with_claude(combined_text)

        # Save results
        output_file = os.path.join(OUTPUT_DIR, f"{sku}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(attributes_json)

        product["status"] = "done"
        product["attributes_file"] = output_file
        product["timestamp"] = datetime.utcnow().isoformat()


def main():
    # Initialize logging
    logger = setup_logging(OUTPUT_DIR)
    
    # Validate data source path is provided
    if not DATA_SOURCE_PATH:
        raise RuntimeError("DATA_SOURCE_PATH is not set. Please set it in environment or .env file.")
    
    logger.info(f"\n📂 Reading data from: {DATA_SOURCE_PATH}")
    logger.info(f"📊 Data source type: {'Excel (SharePoint)' if USE_EXCEL_DATA else 'CSV (Local)'}\n")
    
    # Read product data from Excel or CSV
    all_products = read_rows(DATA_SOURCE_PATH)
    logger.info(f"✅ Loaded {len(all_products)} products from data source\n")
    
    # Resolve registry path: allow REGISTRY_PATH to be a directory or a file
    registry_path_cfg = REGISTRY_PATH
    if os.path.isdir(registry_path_cfg):
        registry_file = os.path.join(registry_path_cfg, "registry.json")
    else:
        # If a file path was provided (possibly without extension), use as-is
        registry_file = registry_path_cfg
        # If it ends with a path separator for some reason, still fall back to registry.json
        if registry_file.endswith(os.sep):
            registry_file = os.path.join(registry_file, "registry.json")

    # Ensure parent directory exists
    parent_dir = os.path.dirname(registry_file) or "."
    os.makedirs(parent_dir, exist_ok=True)

    # Load or initialize registry for progress tracking
    if os.path.exists(registry_file) and os.path.isfile(registry_file):
        with open(registry_file, "r", encoding="utf-8") as f:
            try:
                registry = json.load(f)
            except Exception:
                registry = []
        logger.info(f"📋 Loaded existing registry with {len(registry)} entries from {registry_file}")
    else:
        registry = []
        logger.info(f"📋 No existing registry found, starting fresh at {registry_file}")
    
    # Create a lookup of completed SKUs from registry
    completed_skus = {p.get("sku") for p in registry if p.get("status") == "done" and p.get("sku")}
    logger.info(f"✓ {len(completed_skus)} products already completed\n")
    
    # Filter out already completed products
    pending = [p for p in all_products if p.get("sku") and p.get("sku") not in completed_skus]
    logger.info(f"🔄 {len(pending)} products pending processing\n")
    
    if not pending:
        logger.info("✅ All products have already been processed!")
        return
    
    # Use Anthropic Messages Batch API for processing
    logger.info("\n🚀 Starting Anthropic batch processing for %s products...", len(pending))
    ok, fail = run_batch_attributes(
        pending,
        model="claude-opus-4-1-20250805",
        max_tokens=31000,
        batch_size=BATCH_SIZE,
        use_cache=True,
        output_dir=OUTPUT_DIR,
        logger=logger,
    )

    # Merge statuses back into registry
    for product in pending:
        existing = next((p for p in registry if p.get("sku") == product.get("sku")), None)
        if existing:
            existing.update(product)
        else:
            registry.append(product)

    # Save progress
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    logger.info("💾 Progress saved to registry")
    logger.info("\n✅ Batch processing complete. ok=%s fail=%s", ok, fail)

    # Optionally trigger auto-upserting to Pinecone (index 'common-attributes')
    if RUN_COMMON_ATTR_UPSERT:
        try:
            logger.info("\n▶️ Auto-upserting generated attributes to Pinecone (index 'common-attributes')...")
            from attribute_generation_upsert.core.common_and_max_attributes_upserting import main as upsert_common_attrs_main
            upsert_common_attrs_main()
            logger.info("✅ Auto-upsert completed.")
        except Exception as e:
            logger.error("❌ Auto-upsert failed: %s", e)


if __name__ == "__main__":
    main()
