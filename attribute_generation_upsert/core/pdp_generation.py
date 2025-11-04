import os
import json
import csv
import sys
import logging
import argparse
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
# CSV conversion helpers (robust import paths)
try:
    # 1) Try package-qualified import
    from attribute_generation_upsert.core.json_to_csv_pdp import (  # type: ignore
        json_obj_to_csv,
        extract_first_json_object,
    )
except Exception:
    try:
        # 2) Try local module import when running from core/ directly
        from json_to_csv_pdp import json_obj_to_csv, extract_first_json_object  # type: ignore
    except Exception:
        try:
            # 3) Inject project root to sys.path and retry package import
            _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if _proj_root not in sys.path:
                sys.path.insert(0, _proj_root)
            from attribute_generation_upsert.core.json_to_csv_pdp import (  # type: ignore
                json_obj_to_csv,
                extract_first_json_object,
            )
        except Exception:
            json_obj_to_csv = None  # type: ignore
            extract_first_json_object = None  # type: ignore
            # Defer logging until logger is configured; later calls will log a debug message when None

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
PDP_LOGS_DIR = os.getenv(
    "PDP_LOGS_DIR",
    "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/logs_pdp",
).strip()

# Prompt file (JSON) containing a "prompt" key
PROMPT_PATH = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/data/FINAL_V22_expert_extraction_prompt.json"
# Data source configuration
USE_EXCEL_DATA = True  # Set to False to use CSV
DATA_SOURCE_PATH = "https://kult20256-my.sharepoint.com/:x:/g/personal/harshit_a_kult_app/ER4mU46_r9VAu0XCFYkVzD8Be1E4BFyHPulmXQYVf0ZTtQ?rtime=Yly5qN0K3kg"

BATCH_SIZE = 5
REGISTRY_PATH = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/checkpoint"
OUTPUT_DIR = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/pdp_claude_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV output directory for per-product CSV alongside automation
CSV_OUTPUT_DIR = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/csv_pdp_output"
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Optional: automatically upsert generated JSONs to Pinecone after batch finishes
# Set to True to enable auto-upsert via common_and_max_attributes_upserting.py
RUN_COMMON_ATTR_UPSERT = False

# Enable extended thinking for Claude
ENABLE_THINKING = True

# Enable file-based locking to avoid duplicate processing across concurrent runs
USE_LOCKS = False

# Single source of truth for Claude model and token budget
DEFAULT_MODEL = "claude-opus-4-1-20250805"
MAX_TOKENS_DEFAULT = 32000

# Google Drive upload integration (disabled by default)
ENABLE_DRIVE_UPLOAD: bool = False
try:
    # Make QnA_Generation importable to reuse the same uploader util
    _base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "QnA_Generation"))
    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)
    from google_drive_uploader import upload_file_to_drive, TARGET_FOLDER_ID  # type: ignore
except Exception:
    # Uploader may not be available in some environments; keep feature optional
    upload_file_to_drive = None  # type: ignore
    TARGET_FOLDER_ID = None  # type: ignore

# Optional override: specify Drive folder ID here; takes precedence over imported TARGET_FOLDER_ID
# Set from user-provided folder link
DRIVE_FOLDER_ID: str | None = "10sxNHyZK7WuEpq6QkeUdmoY-ecmp89_E"

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in environment/.env")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set in environment/.env")
if not INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME is not set in environment/.env")

# =============================
# 🧭 CATEGORY/CHECKPOINT CONFIG (aligned with QnA_main)
# =============================

# Category-scoped checkpointing (update as needed before execution)
DEFAULT_CATEGORY: str = "Lipstick_pdp_saas"

# Centralized checkpoint directory (shared approach)
CHECKPOINT_BASE_DIR: str = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/checkpoint"


def _sanitize_category(cat: str) -> str:
    """Sanitize category to a filename-safe token."""
    s = (cat or "").strip()
    if not s:
        return "uncategorized"
    safe = []
    for c in s:
        if c.isalnum():
            safe.append(c)
        elif c in (" ", "-", "_", "/"):
            safe.append("_")
        else:
            safe.append("")
    out = "".join(safe).strip("_") or "uncategorized"
    return out


def get_attributes_checkpoint_path() -> str:
    """Path to the centralized, category-scoped attributes checkpoint file."""
    category_safe = _sanitize_category(DEFAULT_CATEGORY)
    path = os.path.join(CHECKPOINT_BASE_DIR, f"checkpoint_{category_safe}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# =============================
# 💾 Attributes checkpoint helpers (modeled after QnA_main)
# =============================

def load_attributes_checkpoint() -> Dict[str, Any]:
    """Load attributes checkpoint data from centralized, category-scoped file."""
    checkpoint_file = get_attributes_checkpoint_path()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data.setdefault("completed_products", [])
                data.setdefault("completed_skus", [])
                data.setdefault("failed_products", [])
                data.setdefault("failed_skus", [])
                data.setdefault("total_processed", 0)
                data.setdefault("successful", 0)
                data.setdefault("failed", 0)
                data.setdefault("last_updated", None)
                data.setdefault("session_stats", [])
                if isinstance(data.get("completed_skus"), list):
                    data["completed_skus"] = set(data["completed_skus"])
                if isinstance(data.get("failed_skus"), list):
                    data["failed_skus"] = set(data["failed_skus"])
                return data
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load attributes checkpoint: {e}")
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


def save_attributes_checkpoint(checkpoint_data: Dict[str, Any]) -> None:
    """Save attributes checkpoint data to centralized, category-scoped file."""
    checkpoint_file = get_attributes_checkpoint_path()
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
        logging.getLogger(__name__).error(f"Failed to save attributes checkpoint: {e}")


def _compose_product_id(product_row: Dict[str, str]) -> str:
    brand = product_row.get('brand', '')
    name = product_row.get('product_name', '')
    shade = product_row.get('shade_of_lipstick', '')
    pid = f"{brand}_{name}_{shade}"
    if product_row.get('sku'):
        pid += f"_{product_row['sku']}"
    return pid


def update_attributes_checkpoint(product_row: Dict[str, str], success: bool, checkpoint_data: Dict[str, Any]) -> None:
    """Update attributes checkpoint with product completion/failure and persist."""
    product_id = _compose_product_id(product_row)
    if success:
        if product_id not in checkpoint_data.get("completed_products", []):
            checkpoint_data.setdefault("completed_products", []).append(product_id)
        if product_id in checkpoint_data.get("failed_products", []):
            checkpoint_data["failed_products"].remove(product_id)
        if product_row.get('sku'):
            checkpoint_data.setdefault("completed_skus", set()).add(product_row['sku'])
            checkpoint_data.setdefault("failed_skus", set()).discard(product_row['sku'])
    else:
        if product_id not in checkpoint_data.get("failed_products", []):
            checkpoint_data.setdefault("failed_products", []).append(product_id)
        if product_id in checkpoint_data.get("completed_products", []):
            checkpoint_data["completed_products"].remove(product_id)
        if product_row.get('sku'):
            checkpoint_data.setdefault("failed_skus", set()).add(product_row['sku'])
            checkpoint_data.setdefault("completed_skus", set()).discard(product_row['sku'])

    completed_skus = checkpoint_data.get("completed_skus", set())
    failed_skus = checkpoint_data.get("failed_skus", set())
    completed_products = checkpoint_data.get("completed_products", [])
    failed_products = checkpoint_data.get("failed_products", [])

    if failed_products:
        checkpoint_data["failed_products"] = [pid for pid in failed_products if pid not in set(completed_products)]

    if isinstance(completed_skus, set) and isinstance(failed_skus, set) and (len(completed_skus) + len(failed_skus)) > 0:
        checkpoint_data["successful"] = len(completed_skus)
        checkpoint_data["failed"] = len(failed_skus)
        checkpoint_data["total_processed"] = checkpoint_data["successful"] + checkpoint_data["failed"]
    else:
        checkpoint_data["successful"] = len(set(completed_products))
        checkpoint_data["failed"] = len(set(checkpoint_data.get("failed_products", [])))
        checkpoint_data["total_processed"] = checkpoint_data["successful"] + checkpoint_data["failed"]

    save_attributes_checkpoint(checkpoint_data)


def is_product_completed_attr(product_row: Dict[str, str], checkpoint_data: Dict[str, Any]) -> bool:
    """Check if product is already completed according to attributes checkpoint."""
    if product_row.get('sku') and product_row['sku'] in checkpoint_data.get("completed_skus", set()):
        return True
    product_id = _compose_product_id(product_row)
    return product_id in checkpoint_data.get("completed_products", [])


def is_product_failed_attr(product_row: Dict[str, str], checkpoint_data: Dict[str, Any]) -> bool:
    """Check if product is currently marked as failed in attributes checkpoint."""
    if product_row.get('sku') and product_row['sku'] in checkpoint_data.get("failed_skus", set()):
        return True
    product_id = _compose_product_id(product_row)
    return product_id in checkpoint_data.get("failed_products", [])


def is_pdp_completed(sku: str, checkpoint_data: Dict[str, Any]) -> bool:
    """Convenience: check if PDP generation is completed for a SKU (uses attributes checkpoint)."""
    return sku in checkpoint_data.get("completed_skus", set())


# ------------------------------
# Batch Tracking System (for visibility + resume support)
# ------------------------------

def get_batch_tracker_path() -> str:
    """Get path to the category-scoped batch tracker file for PDP pipeline."""
    category_safe = _sanitize_category(DEFAULT_CATEGORY)
    # Use a pdp-specific name to avoid colliding with QnA's tracker
    path = os.path.join(CHECKPOINT_BASE_DIR, f"batch_tracker_pdp_{category_safe}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_batch_tracker(output_dir: str) -> Dict[str, Any]:
    """Load batch tracker data from centralized, category-scoped file.
    Note: output_dir is ignored; kept for signature parity.
    """
    tracker_file = get_batch_tracker_path()
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load batch tracker: {e}")
    return {
        "pending_batches": [],  # list of {batch_id, submitted_at, wave_start, wave_end, rows: [...]}
        "completed_batches": [],
        "last_updated": None,
    }


def save_batch_tracker(output_dir: str, tracker_data: Dict[str, Any]) -> None:
    """Save batch tracker data to centralized, category-scoped file.
    Note: output_dir is ignored; kept for signature parity.
    """
    tracker_file = get_batch_tracker_path()
    tracker_data["last_updated"] = datetime.now().isoformat()
    try:
        with open(tracker_file, "w", encoding="utf-8") as f:
            json.dump(tracker_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save batch tracker: {e}")


def add_pending_batch(output_dir: str, batch_id: str, wave_start: int, wave_end: int, rows: List[Dict[str, str]]) -> None:
    """Register a newly submitted batch as pending (category-scoped)."""
    tracker = load_batch_tracker(output_dir)
    # Serialize minimal fields for resume/debug
    rows_minimal = []
    for r in rows:
        rows_minimal.append({
            "custom_id": _make_custom_id(r),
            "brand": r.get("brand", ""),
            "product_name": r.get("product_name", ""),
            "shade_of_lipstick": r.get("shade_of_lipstick", ""),
            "sku": r.get("sku", ""),
            "category": r.get("category", ""),
            "sub_category": r.get("sub_category", ""),
            "leaf_level_category": r.get("leaf_level_category", ""),
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
    """Move a batch from pending to completed in the tracker."""
    tracker = load_batch_tracker(output_dir)
    pending_batch = None
    for i, batch in enumerate(tracker.get("pending_batches", [])):
        if batch.get("batch_id") == batch_id:
            pending_batch = tracker["pending_batches"].pop(i)
            break
    if pending_batch:
        pending_batch["completed_at"] = datetime.now().isoformat()
        tracker.setdefault("completed_batches", []).append(pending_batch)
        save_batch_tracker(output_dir, tracker)

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
    # Use dedicated logs directory (overridable via PDP_LOGS_DIR env var)
    log_dir = PDP_LOGS_DIR if PDP_LOGS_DIR else os.path.join(output_dir, "logs")
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
            Return ONLY valid JSON (no markdown, no commentary)
            """
        ).strip() + "\n",
    }

    big_block = {
        "type": "text",
        "text": textwrap.dedent(
            f"""
            {prompt_text}

            CRITICAL FORMATTING REQUIREMENTS:
            - Return ONLY valid JSON (no markdown, no commentary)
            - Do NOT include code fences (```), markdown, prose, or any text before/after the JSON
            - Start your response with '{{' and return exactly one complete JSON object (no arrays, no multiple objects).
            - Absolutely remove any markdown links or inline URLs from string values. If a value contains a pattern like [Text](https://example.com) or (https://example.com), keep only the plain text 'Text' and drop the URL entirely.
            - Ensure all internal double quotes inside JSON string values are escaped with a backslash. For example, if a value needs quotes like He said "Burnt Pumpkin", output "He said \\"Burnt Pumpkin\\"" in JSON.
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


# =============================
# 🧰 OUTPUT/LOCK HELPERS
# =============================

def _output_file_for_sku(sku: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{sku}.json")


def output_exists(sku: str) -> bool:
    path = _output_file_for_sku(sku)
    return os.path.exists(path) and os.path.getsize(path) > 0


LOCKS_DIR = os.path.join(OUTPUT_DIR, ".locks")
os.makedirs(LOCKS_DIR, exist_ok=True)


def _lock_path_for_sku(sku: str) -> str:
    safe = "".join(c for c in sku if c.isalnum() or c in ("-", "_"))
    return os.path.join(LOCKS_DIR, f"{safe}.lock")


def is_locked(sku: str, *, fresh_seconds: int = 6 * 3600) -> bool:
    """Return True if a fresh lock file exists for this SKU (indicating in-progress elsewhere)."""
    if not USE_LOCKS:
        return False
    lp = _lock_path_for_sku(sku)
    if not os.path.exists(lp):
        return False
    try:
        mtime = os.path.getmtime(lp)
        import time as _t
        return (_t.time() - mtime) < fresh_seconds
    except Exception:
        return True


def acquire_lock(sku: str) -> None:
    try:
        if not USE_LOCKS:
            return
        with open(_lock_path_for_sku(sku), "w", encoding="utf-8") as f:
            f.write(datetime.utcnow().isoformat())
    except Exception:
        pass


def release_lock(sku: str) -> None:
    try:
        if not USE_LOCKS:
            return
        lp = _lock_path_for_sku(sku)
        if os.path.exists(lp):
            os.remove(lp)
    except Exception:
        pass


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
            top_k=10,  # Get all relevant chunks
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
        temperature=1,
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
    model: str | None = None,
    max_tokens: int | None = None,
    batch_size: int | None = None,
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

    # Resolve defaults from single source of truth
    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or MAX_TOKENS_DEFAULT
    batch_size = batch_size or BATCH_SIZE

    # Build shared system blocks once (cacheable)
    shared_system = _build_shared_system_blocks(ATTRIBUTE_PROMPT_TEMPLATE, use_cache)
    try:
        logger.info("Prompt caching (batch): %s", "ENABLED" if use_cache else "DISABLED")
    except Exception:
        pass

    ok, fail = 0, 0
    # Drive upload counters (best-effort)
    upload_success, upload_failed, upload_skipped = 0, 0, 0

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
                "temperature": 1,
                "system": shared_system,
                "messages": [{"role": "user", "content": content}],
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
                requests_payload.append({"custom_id": custom_id, "params": params})

        # Submit batch
        logger.info("[batch][attr] Submitting wave %s: %s requests", i//batch_size + 1, len(requests_payload))
        batch = client.messages.batches.create(requests=requests_payload)
        batch_id = getattr(batch, "id", None) or batch["id"]  # type: ignore
        logger.info("[batch][attr] Batch submitted: id=%s", batch_id)

        # Track as pending
        try:
            wave_start = i
            wave_end = i + len(wave) - 1
            add_pending_batch(output_dir, batch_id, wave_start, wave_end, wave)
        except Exception:
            logger.debug("Failed to record pending batch in tracker: %s", batch_id)

        # Poll until terminal status, with capped wait like QnA_main
        import time as _time
        start_poll = _time.time()
        max_wait_seconds = 3 * 60  # 3 minutes
        timed_out = False
        while True:
            status_obj = client.messages.batches.retrieve(batch_id)
            status = getattr(status_obj, "processing_status", None)
            logger.info("[batch][attr] Batch %s status: %s", batch_id, status)
            if status is None:
                logger.warning("[batch][attr] Batch %s status response missing processing_status: %r", batch_id, status_obj)
                break
            if status in ("ended", "completed", "cancelled", "expired", "failed"):
                break
            elapsed = _time.time() - start_poll
            if elapsed >= max_wait_seconds:
                logger.warning("[batch][attr] Batch %s polling timed out after %.1f minutes; deferring results to resume.", batch_id, elapsed / 60.0)
                timed_out = True
                break
            _time.sleep(3)

        # If timed out, skip fetching results now; resume_pending_batches_pdp will handle it
        if timed_out:
            logger.info("[batch][attr] Deferring results retrieval for batch %s to resume phase; moving on to next wave.", batch_id)
            continue

        # Retrieve results now
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
                # Mark status as failed for registry propagation
                try:
                    row = next((r for r in wave if _make_custom_id(r) == custom_id), None)
                    if row is not None:
                        row["status"] = "failed"
                        row["timestamp"] = datetime.utcnow().isoformat()
                except Exception:
                    pass
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

            # Also save CSV derived from JSON (best-effort, non-blocking)
            try:
                if json_obj_to_csv is not None:
                    # Parse JSON robustly from the text payload
                    parsed = None
                    try:
                        parsed = json.loads(text_payload)
                    except Exception:
                        if extract_first_json_object is not None:
                            candidate = extract_first_json_object(text_payload)
                            if candidate:
                                parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        csv_path = os.path.join(CSV_OUTPUT_DIR, f"{sku}.csv")
                        json_obj_to_csv(parsed, csv_path)
                        logger.info("[csv] Saved CSV for %s at %s", sku, csv_path)
                    else:
                        logger.warning("[csv] Could not parse JSON payload for %s; CSV not created", sku)
                else:
                    logger.debug("[csv] json_to_csv_pdp helpers not available; skipping CSV generation")
            except Exception as e_csv:
                logger.warning("[csv] Failed generating CSV for %s: %s", sku, e_csv)

            # Optional: upload the generated JSON to Google Drive BEFORE marking as done
            uploaded_ok = True  # default to true when uploads are disabled
            if ENABLE_DRIVE_UPLOAD:
                uploaded_ok = False
                _folder_id = DRIVE_FOLDER_ID or TARGET_FOLDER_ID
                if upload_file_to_drive and _folder_id:
                    try:
                        up_ok = upload_file_to_drive(
                            file_path=output_file,
                            folder_id=_folder_id,
                            logger=logger,
                        )
                        if not up_ok:
                            logger.warning("[drive] Upload failed for %s", output_file)
                            upload_failed += 1
                        else:
                            logger.info("[drive] Uploaded %s to Drive folder %s", os.path.basename(output_file), _folder_id)
                            upload_success += 1
                            uploaded_ok = True
                    except Exception as e:
                        logger.error("[drive] Exception uploading %s: %s", output_file, e)
                        upload_failed += 1
                else:
                    logger.info("[drive] Drive upload disabled or uploader not available")
                    upload_skipped += 1

            if uploaded_ok:
                # Update row status so caller can update registry
                row["status"] = "done"
                row["attributes_file"] = output_file
                row["timestamp"] = datetime.utcnow().isoformat()
                ok += 1
            else:
                # Mark as pending upload; do NOT count as completed
                row["status"] = "pending_upload"
                row["attributes_file"] = output_file
                row["timestamp"] = datetime.utcnow().isoformat()

        # Mark batch completed in tracker
        try:
            mark_batch_completed(output_dir, batch_id)
        except Exception:
            logger.debug("Failed to mark batch completed in tracker: %s", batch_id)

    # Log upload summary (only when feature is enabled)
    try:
        if ENABLE_DRIVE_UPLOAD:
            logger.info("[drive][summary] uploaded=%s failed=%s skipped=%s", upload_success, upload_failed, upload_skipped)
    except Exception:
        pass

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


def resume_pending_batches_pdp(
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    output_dir: str,
    logger: logging.Logger,
    use_cache: bool = True,
) -> Tuple[int, int]:
    """Resume processing of pending Anthropic batches recorded in the tracker.
    Simpler variant without pause_turn handling.
    Returns (ok_count, fail_count).
    """
    # Defaults
    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or MAX_TOKENS_DEFAULT

    tracker = load_batch_tracker(output_dir)
    pending_batches = tracker.get("pending_batches", [])
    if not pending_batches:
        try:
            logger.info("[resume] No pending batches found")
        except Exception:
            pass
        return 0, 0

    try:
        logger.info("[resume] Found %s pending batches to process", len(pending_batches))
    except Exception:
        pass

    client_local = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    shared_system = _build_shared_system_blocks(ATTRIBUTE_PROMPT_TEMPLATE or read_prompt(PROMPT_PATH), use_cache)

    # Load attributes checkpoint once
    checkpoint_data = load_attributes_checkpoint()

    ok, fail = 0, 0

    for batch_info in list(pending_batches):
        batch_id = batch_info.get("batch_id")
        rows_minimal = batch_info.get("rows", [])
        try:
            logger.info("[resume] Processing batch %s (%s products)", batch_id, len(rows_minimal))
        except Exception:
            pass

        try:
            # Check status (best-effort brief wait if still running)
            status_obj = client_local.messages.batches.retrieve(batch_id)
            status = getattr(status_obj, "processing_status", None)
            logger.info("[resume] Batch %s current status: %s", batch_id, status)
            if status not in ("ended", "completed", "cancelled", "expired", "failed"):
                import time as _t
                logger.info("[resume] Waiting briefly for batch %s to finish...", batch_id)
                start_wait = _t.time()
                while _t.time() - start_wait < 300:  # wait up to 2 minutes
                    _t.sleep(5)
                    status_obj = client_local.messages.batches.retrieve(batch_id)
                    status = getattr(status_obj, "processing_status", None)
                    logger.info("[resume] Batch %s status: %s", batch_id, status)
                    if status in ("ended", "completed", "cancelled", "expired", "failed"):
                        break

            # Fetch results
            results_iter = client_local.messages.batches.results(batch_id)
            try:
                items = list(results_iter)
            except Exception:
                items = getattr(results_iter, "data", None) or results_iter.get("data", [])  # type: ignore
            logger.info("[resume] Batch %s returned %s results", batch_id, len(items))

            # Map custom_id to row for checkpoint and naming
            row_map = {(_make_custom_id(r)): r for r in rows_minimal}

            for item in items:
                custom_id = getattr(item, "custom_id", None) or item.get("custom_id")  # type: ignore
                result_obj = getattr(item, "result", None) or item.get("result")  # type: ignore
                if getattr(result_obj, "type", None) != "succeeded":
                    # Mark failure in attributes checkpoint (best-effort)
                    r_min = row_map.get(custom_id)
                    if r_min:
                        update_attributes_checkpoint(r_min, False, checkpoint_data)
                    fail += 1
                    err_wrapper = getattr(result_obj, "error", None)
                    err_inner = getattr(err_wrapper, "error", None)
                    err_msg = (
                        getattr(err_inner, "message", None)
                        or getattr(err_wrapper, "message", None)
                        or None
                    )
                    if err_msg:
                        logger.error("[resume] Item failed: %s -> %s", custom_id, err_msg)
                    continue

                message = getattr(result_obj, "message", None)
                r_min = row_map.get(custom_id)
                if not r_min:
                    logger.warning("[resume] Result custom_id=%s has no matching row", custom_id)
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
                            "[resume][cache] custom_id=%s status=%s token_credits=%s", custom_id, cache_status, cache_token_credits
                        )
                    else:
                        logger.info("[resume][cache] custom_id=%s status=%s", custom_id, None)
                except Exception:
                    logger.debug("[resume] Failed to read cache headers for custom_id=%s", custom_id)

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
                    logger.info("[resume][usage] custom_id=%s in=%s out=%s cache_read_in=%s", custom_id, in_tok, out_tok, cache_read_tok)
                except Exception:
                    pass

                # Extract final text as in main batch (concatenate all text parts)
                try:
                    content_blocks = getattr(message, "content", None) or []
                    parts = []
                    for c in content_blocks:
                        if getattr(c, "type", None) == "text":
                            parts.append(getattr(c, "text", ""))
                    # Use the entire concatenated text, not just the last block
                    text_payload = "".join(parts)
                except Exception:
                    text_payload = ""

                # Write output file
                sku = (r_min.get("sku") or "").strip() or custom_id or "unknown"
                outpath = os.path.join(output_dir, f"{sku}.json")
                try:
                    with open(outpath, "w", encoding="utf-8") as f:
                        f.write(text_payload)
                except Exception as e_w:
                    logger.error("[resume] Failed writing output for %s: %s", sku, e_w)
                    update_attributes_checkpoint(r_min, False, checkpoint_data)
                    fail += 1
                    continue

                # Also save CSV derived from JSON (best-effort)
                try:
                    if json_obj_to_csv is not None:
                        parsed = None
                        try:
                            parsed = json.loads(text_payload)
                        except Exception:
                            if extract_first_json_object is not None:
                                candidate = extract_first_json_object(text_payload)
                                if candidate:
                                    parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            csv_path = os.path.join(CSV_OUTPUT_DIR, f"{sku}.csv")
                            json_obj_to_csv(parsed, csv_path)
                            logger.info("[resume][csv] Saved CSV for %s at %s", sku, csv_path)
                        else:
                            logger.warning("[resume][csv] Could not parse JSON payload for %s; CSV not created", sku)
                    else:
                        logger.debug("[resume][csv] json_to_csv_pdp helpers not available; skipping CSV generation")
                except Exception as e_csv:
                    logger.warning("[resume][csv] Failed generating CSV for %s: %s", sku, e_csv)

                # Optional Drive upload
                if ENABLE_DRIVE_UPLOAD:
                    _folder_id = DRIVE_FOLDER_ID or TARGET_FOLDER_ID
                    if upload_file_to_drive and _folder_id:
                        try:
                            _ok = upload_file_to_drive(file_path=outpath, folder_id=_folder_id, logger=logger)
                            if not _ok:
                                logger.warning("[resume] Drive upload failed for %s", os.path.basename(outpath))
                        except Exception as e_up:
                            logger.error("[resume] Drive upload exception for %s: %s", os.path.basename(outpath), e_up)
                    else:
                        logger.info("[resume] Drive upload disabled or uploader not available")

                # Success
                update_attributes_checkpoint(r_min, True, checkpoint_data)
                ok += 1

            # Mark batch completed in tracker
            mark_batch_completed(output_dir, batch_id)
            logger.info("[resume] Batch %s marked completed", batch_id)

        except Exception as e:
            logger.error("[resume] Error processing batch %s: %s", batch_id, e)
            # Mark all rows as failed
            for r_min in rows_minimal:
                update_attributes_checkpoint(r_min, False, checkpoint_data)
            fail += len(rows_minimal)

    return ok, fail


def main():
    # Initialize logging
    logger = setup_logging(OUTPUT_DIR)

    # CLI args: add resume mode to process pending batches and exit
    parser = argparse.ArgumentParser(description="PDP generation runner")
    parser.add_argument("--resume_batches", action="store_true", help="Resume processing of pending batches that timed out during polling")
    args, _unknown = parser.parse_known_args()
    try:
        logging.getLogger(__name__).info("Parsed flags: resume_batches=%s", args.resume_batches)
    except Exception:
        pass

    # If resume mode, process pending and exit early
    if getattr(args, "resume_batches", False):
        logger.info("\n🔄 RESUME MODE: Processing pending PDP batches...")
        logger.info("=" * 60)
        try:
            r_ok, r_fail = resume_pending_batches_pdp(
                model=None,
                max_tokens=None,
                output_dir=OUTPUT_DIR,
                logger=logger,
                use_cache=True,
            )
            logger.info("[resume] Completed. ok=%s fail=%s", r_ok, r_fail)
        except Exception as e:
            logger.error("[resume] Failed: %s", e)
        return
    # Load centralized, category-scoped attributes checkpoint
    attr_checkpoint = load_attributes_checkpoint()
    
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
    category_safe = _sanitize_category(DEFAULT_CATEGORY)
    if os.path.isdir(registry_path_cfg) or registry_path_cfg.endswith(os.sep):
        # Directory case: write category-scoped registry file
        registry_file = os.path.join(registry_path_cfg, f"registry_{category_safe}.json")
    else:
        # File path provided
        base = os.path.basename(registry_path_cfg)
        parent = os.path.dirname(registry_path_cfg) or "."
        if base.lower().endswith(".json"):
            name_no_ext, ext = os.path.splitext(base)
            # If named generic 'registry.json', make it category-scoped
            if name_no_ext == "registry":
                base = f"registry_{category_safe}{ext}"
            registry_file = os.path.join(parent, base)
        else:
            # Treat as directory-like path sans trailing slash
            registry_file = os.path.join(registry_path_cfg, f"registry_{category_safe}.json")

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
    
    # Create a lookup of completed SKUs from registry and merge with checkpoint knowledge
    completed_skus = {p.get("sku") for p in registry if p.get("status") == "done" and p.get("sku")}
    # Merge in completed SKUs tracked by checkpoint
    try:
        completed_skus |= set(attr_checkpoint.get("completed_skus", set()))
    except Exception:
        pass
    logger.info(f"✓ {len(completed_skus)} products already completed\n")

    # Load tracker and build a set of SKUs already in pending batches to avoid duplicate LLM calls
    try:
        tracker = load_batch_tracker(OUTPUT_DIR)
        pending_tracker_skus = set()
        for _b in tracker.get("pending_batches", []):
            for _r in _b.get("rows", []):
                _sku = (_r.get("sku") or "").strip()
                if not _sku:
                    # fallback to custom_id in case sku was empty in tracker
                    _sku = (_r.get("custom_id") or "").strip()
                if _sku:
                    pending_tracker_skus.add(_sku)
        logger.info("⏳ %s products already in pending batches (tracker)", len(pending_tracker_skus))
    except Exception:
        pending_tracker_skus = set()

    # Filter out already completed products; also skip if output file already exists or lock is fresh
    pending: List[Dict[str, Any]] = []
    for p in all_products:
        sku = (p.get("sku") or "").strip()
        if not sku:
            continue
        # Skip if known completed by registry or checkpoint
        if sku in completed_skus or is_product_completed_attr(p, attr_checkpoint):
            try:
                logger.info("🧩 Duplicate: %s already processed via registry/checkpoint; skipping.", sku)
            except Exception:
                pass
            continue
        # If SKU is already recorded in pending batches, skip to prevent duplicate batch submissions
        if sku in pending_tracker_skus:
            logger.info("⏸️  Skipping %s (already recorded in pending batches tracker)", sku)
            continue
        # If output already exists on disk, prefer syncing to Drive first (if enabled), then mark done
        if output_exists(sku):
            # Log skip with file path for visibility
            try:
                _existing_file = _output_file_for_sku(sku)
                logger.info("📁 Output already exists for %s at %s; skipping generation.", sku, _existing_file)
            except Exception:
                _existing_file = _output_file_for_sku(sku)
                logger.info("📁 Output already exists for %s; skipping generation.", sku)

            uploaded_ok = True  # default to true if uploads are disabled
            if ENABLE_DRIVE_UPLOAD:
                uploaded_ok = False
                try:
                    _folder_id = DRIVE_FOLDER_ID or TARGET_FOLDER_ID
                    if upload_file_to_drive and _folder_id:
                        up_ok = upload_file_to_drive(
                            file_path=_existing_file,
                            folder_id=_folder_id,
                            logger=logger,
                            replace_existing=False,
                        )
                        if not up_ok:
                            logger.warning("[sync][existing] Drive upload failed for %s", os.path.basename(_existing_file))
                        else:
                            uploaded_ok = True
                    else:
                        logger.info("[sync][existing] Drive upload disabled or folder id missing")
                except Exception as e:
                    logger.error("[sync][existing] Drive upload exception for %s: %s", os.path.basename(_existing_file), e)

            # Only now mark as completed based on upload policy
            if uploaded_ok:
                existing = next((r for r in registry if r.get("sku") == sku), None)
                if existing:
                    existing.update({"status": "done", "attributes_file": _existing_file, "timestamp": datetime.utcnow().isoformat()})
                else:
                    registry.append({**p, "status": "done", "attributes_file": _existing_file, "timestamp": datetime.utcnow().isoformat()})
                try:
                    update_attributes_checkpoint(p, True, attr_checkpoint)
                except Exception:
                    logger.debug("Failed to update attributes checkpoint for pre-existing output: %s", sku)
            else:
                # Track as pending upload; do NOT mark completed
                existing = next((r for r in registry if r.get("sku") == sku), None)
                entry = {**p, "status": "pending_upload", "attributes_file": _existing_file, "timestamp": datetime.utcnow().isoformat()}
                if existing:
                    existing.update(entry)
                else:
                    registry.append(entry)
            continue
        # If a fresh lock exists, skip (another run is processing it)
        if is_locked(sku):
            logger.info("⏸️  Skipping %s (already processing elsewhere)", sku)
            continue
        pending.append(p)
    logger.info(f"🔄 {len(pending)} products pending processing\n")
    
    if not pending:
        logger.info("✅ All products have already been processed!")
        return
    
    # Mark as processing and acquire lock before submitting
    for p in pending:
        sku = (p.get("sku") or "").strip()
        p["status"] = "processing"
        p["timestamp"] = datetime.utcnow().isoformat()
        acquire_lock(sku)

    # Persist intermediate registry state
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    # Use Anthropic Messages Batch API for processing
    logger.info("\n🚀 Starting Anthropic batch processing for %s products...", len(pending))
    ok, fail = run_batch_attributes(
        pending,
        model=None,
        max_tokens=None,
        batch_size=None,
        use_cache=True,
        output_dir=OUTPUT_DIR,
        logger=logger,
    )

    # Note: Do not auto-resume pending batches here. Pending batches should be
    # explicitly resumed only when running with the --resume_batches flag.

    # Release locks after batch
    for p in pending:
        sku = (p.get("sku") or "").strip()
        release_lock(sku)

    # Merge statuses back into registry and update attributes checkpoint based on results
    for product in pending:
        existing = next((p for p in registry if p.get("sku") == product.get("sku")), None)
        if existing:
            existing.update(product)
        else:
            registry.append(product)
        # Update checkpoint for each processed product
        try:
            success = (product.get("status") == "done")
            update_attributes_checkpoint(product, success, attr_checkpoint)
        except Exception:
            logger.debug("Failed to update attributes checkpoint for %s", product.get("sku"))

    # Save progress
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    logger.info("💾 Progress saved to registry")
    logger.info("\n✅ Batch processing complete. ok=%s fail=%s", ok, fail)


if __name__ == "__main__":
    main()
