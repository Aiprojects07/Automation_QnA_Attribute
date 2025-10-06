"""
Search and Answer core utilities with Excel integration

This module centralizes configuration and helpers for:
- Anthropic client (Claude Sonnet 4)
- Prompt file path (path only; no prompt content here)
- Pinecone index name
- OpenAI embeddings for query vectorization
- Excel integration for SharePoint URLs and attribute extraction

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
from typing import List, Optional, Sequence, Dict, Any
from urllib.parse import urlsplit, quote as _urlquote
from io import BytesIO
import requests
import pandas as pd

try:
    # Optional: if python-dotenv is installed and a .env exists, load it.
    # Safe if missing; the rest of the code relies on environment variables.
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    # Optional: Google Sheets integration
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

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
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "lipstick-cluster")
# Pinecone namespace (env override supported)
PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "default")

# Retrieval size (centralized)
DEFAULT_TOP_K: int = 5

# CLI-independent defaults
DEFAULT_CSV_PATH: str = \
    "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/core/lipstick_list.csv"
# Google Sheets configuration
GOOGLE_SHEETS_CREDENTIALS_PATH: str = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "")
GOOGLE_SHEETS_NAME: str = os.getenv("GOOGLE_SHEETS_NAME", "Lipstick Attributes")
GOOGLE_SHEETS_WORKSHEET: str = os.getenv("GOOGLE_SHEETS_WORKSHEET", "Sheet1")

# Processing configuration
FORCE_REPROCESS: bool = False
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


def read_excel_from_url(excel_url: str) -> pd.DataFrame:
    """Read Excel file from SharePoint URL and return as DataFrame."""
    download_url = convert_sharepoint_url(excel_url)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AttributeExtractor/1.0)"}
    
    response = requests.get(download_url, headers=headers, timeout=30)
    if not response.ok:
        if response.status_code == 404 and 'download.aspx' in download_url:
            alt_url = download_url.replace('download.aspx', 'guestaccess.aspx')
            response = requests.get(alt_url, headers=headers, timeout=30)
        if not response.ok:
            response.raise_for_status()
    
    excel_data = BytesIO(response.content)
    df = pd.read_excel(excel_data, engine='openpyxl')
    return df


def save_excel_to_url(df: pd.DataFrame, excel_url: str) -> None:
    """Save DataFrame back to Excel file. Note: This requires write access to SharePoint."""
    # For SharePoint, we'll save locally and provide instructions
    # Direct SharePoint write requires more complex authentication
    local_path = "/tmp/updated_attributes.xlsx"
    df.to_excel(local_path, index=False, engine='openpyxl')
    print(f"Excel file saved locally to: {local_path}")
    print("Please upload this file back to SharePoint manually or set up SharePoint write access.")


def save_to_excel_file(df: pd.DataFrame, output_path: str = None) -> str:
    """Save DataFrame to local Excel file."""
    if output_path is None:
        output_path = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/output/attributes_output.xlsx"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    return output_path




def read_google_sheets(credentials_path: str = None, sheet_name: str = None, worksheet: str = None) -> pd.DataFrame:
    """Read data from Google Sheets and return as DataFrame."""
    if not GSPREAD_AVAILABLE:
        raise ImportError("gspread and google-auth packages are required for Google Sheets integration")
    
    creds_path = credentials_path or GOOGLE_SHEETS_CREDENTIALS_PATH
    sheet_name = sheet_name or GOOGLE_SHEETS_NAME
    worksheet_name = worksheet or GOOGLE_SHEETS_WORKSHEET
    
    if not creds_path or not os.path.exists(creds_path):
        raise FileNotFoundError(f"Google Sheets credentials file not found: {creds_path}")
    
    # Authenticate and open sheet
    gc = gspread.service_account(filename=creds_path)
    sheet = gc.open(sheet_name).worksheet(worksheet_name)
    
    # Get all records as list of dictionaries
    records = sheet.get_all_records()
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


def save_to_google_sheets(df: pd.DataFrame, credentials_path: str = None, sheet_name: str = None, worksheet: str = None) -> bool:
    """Save DataFrame to Google Sheets."""
    if not GSPREAD_AVAILABLE:
        print("⚠️  Google Sheets packages not available. Install with: pip install gspread google-auth")
        return False
    
    try:
        creds_path = credentials_path or GOOGLE_SHEETS_CREDENTIALS_PATH
        sheet_name = sheet_name or GOOGLE_SHEETS_NAME
        worksheet_name = worksheet or GOOGLE_SHEETS_WORKSHEET
        
        if not creds_path or not os.path.exists(creds_path):
            print(f"⚠️  Google Sheets credentials file not found: {creds_path}")
            print("📝 To set up Google Sheets integration:")
            print("   1. Go to Google Cloud Console")
            print("   2. Create service account and download JSON key")
            print("   3. Share your Google Sheet with the service account email")
            print("   4. Set GOOGLE_SHEETS_CREDENTIALS_PATH environment variable")
            return False
        
        # Authenticate and open sheet
        gc = gspread.service_account(filename=creds_path)
        sheet = gc.open(sheet_name).worksheet(worksheet_name)
        
        # Clear existing data and update with new data
        sheet.clear()
        
        # Convert DataFrame to list of lists (including headers)
        data = [df.columns.tolist()] + df.values.tolist()
        
        # Update sheet with all data
        sheet.update('A1', data)
        
        print(f"✅ Google Sheets updated successfully: {sheet_name} → {worksheet_name}")
        print(f"📊 Uploaded {len(df)} rows with {len(df.columns)} columns")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update Google Sheets: {e}")
        return False


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract JSON object from Claude's response text."""
    response_text = response_text.strip()
    
    # Remove code fences if present
    if response_text.startswith("```"):
        response_text = response_text.strip("`")
        if response_text.lower().startswith("json"):
            response_text = response_text[4:].strip()
    
    # Find JSON object
    start = response_text.find('{')
    if start == -1:
        return {}
    
    # Find matching closing brace
    depth = 0
    for i in range(start, len(response_text)):
        if response_text[i] == '{':
            depth += 1
        elif response_text[i] == '}':
            depth -= 1
            if depth == 0:
                json_str = response_text[start:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
    
    # Fallback: try to parse the entire response
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {}


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


def process_csv_to_excel_attributes(csv_path: str = DEFAULT_CSV_PATH, output_excel: str = None, force_reprocess: bool = None) -> int:
    """Read products from CSV, extract attributes using LLM, and save complete output to Excel with checkpointing.
    
    Parameters:
    - csv_path: Path to input CSV file
    - output_excel: Path to output Excel file
    - force_reprocess: If True, reprocess all products regardless of existing attributes
    """
    import sys
    
    # Use global FORCE_REPROCESS if not explicitly provided
    if force_reprocess is None:
        force_reprocess = FORCE_REPROCESS
    
    # Determine output Excel path
    if output_excel is None:
        output_excel = "/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/output/attributes_output.xlsx"
    
    print(f"📊 Reading CSV file from: {csv_path}")
    try:
        csv_df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(csv_df)} products from CSV")
    except Exception as e:
        print(f"❌ Failed to read CSV file: {e}", file=sys.stderr)
        return 2
    
    if csv_df.empty:
        print("❌ CSV file contains no data", file=sys.stderr)
        return 2
    
    # Force reprocess mode
    if force_reprocess:
        print("🔄 Force reprocess mode enabled - will reprocess all products")
        df = csv_df.copy()
        if 'attributes_json' not in df.columns:
            df['attributes_json'] = ''
        existing_df = None
    else:
        # Check if existing Excel file exists and load it for checkpointing
        existing_df = None
        if os.path.exists(output_excel):
            try:
                existing_df = pd.read_excel(output_excel, engine='openpyxl')
                print(f"📋 Found existing Excel file with {len(existing_df)} rows")
                
                # Merge existing processed data with new CSV data
                # Use a unique identifier to match rows (prefer SKU, fallback to index)
                merge_key = None
                for key_col in ['kult_sku_code', 'sku', 'Kult SKU Code', 'SKU']:
                    if key_col in csv_df.columns and key_col in existing_df.columns:
                        merge_key = key_col
                        break
                
                if merge_key:
                    print(f"🔗 Merging data using key column: {merge_key}")
                    # Merge on the key column, keeping existing attributes_json where available
                    df = csv_df.merge(existing_df[['attributes_json', merge_key]], 
                                    on=merge_key, how='left', suffixes=('', '_existing'))
                    # Use existing attributes_json if available, otherwise empty string
                    df['attributes_json'] = df['attributes_json'].fillna('')
                else:
                    print("⚠️  No common key column found, using position-based merge")
                    # Fallback: merge by position (less reliable but better than nothing)
                    df = csv_df.copy()
                    if 'attributes_json' in existing_df.columns and len(existing_df) > 0:
                        # Copy existing attributes_json values by position
                        for i in range(min(len(df), len(existing_df))):
                            if pd.notna(existing_df.iloc[i]['attributes_json']) and str(existing_df.iloc[i]['attributes_json']).strip():
                                df.at[i, 'attributes_json'] = existing_df.iloc[i]['attributes_json']
                    
                    # Add attributes_json column if it doesn't exist
                    if 'attributes_json' not in df.columns:
                        df['attributes_json'] = ''
            except Exception as e:
                print(f"⚠️  Failed to load existing Excel file: {e}")
                print("📊 Starting fresh processing")
                df = csv_df.copy()
                if 'attributes_json' not in df.columns:
                    df['attributes_json'] = ''
        else:
            print("📊 No existing Excel file found, starting fresh")
            df = csv_df.copy()
            # Add attributes_json column if it doesn't exist
            if 'attributes_json' not in df.columns:
                df['attributes_json'] = ''
                print("➕ Added 'attributes_json' column")
    
    # Process each product
    namespace = PINECONE_NAMESPACE
    max_tokens = DEFAULT_MAX_TOKENS
    temperature = DEFAULT_TEMPERATURE
    
    processed_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Skip if already processed (has attributes_json) unless force_reprocess is True
            if not force_reprocess and pd.notna(row.get('attributes_json')) and str(row.get('attributes_json')).strip():
                print(f"⏭️  Skipping row {idx+1}: Already has attributes")
                continue
            elif force_reprocess and pd.notna(row.get('attributes_json')) and str(row.get('attributes_json')).strip():
                print(f"🔄 Force reprocessing row {idx+1}: Overriding existing attributes")
            
            # Normalize column names
            norm_row = {re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_"): str(v).strip() if pd.notna(v) else "" for k, v in row.items()}
            
            # Extract product information
            sku = norm_row.get("kult_sku_code") or norm_row.get("sku") or ""
            brand = norm_row.get("brand") or ""
            product_line = (
                norm_row.get("product_name") or 
                norm_row.get("product") or 
                norm_row.get("product_title") or 
                norm_row.get("line") or ""
            )
            shade = (
                norm_row.get("shade") or 
                norm_row.get("shade_of_lipstick") or 
                norm_row.get("color") or ""
            )
            category = norm_row.get("category") or ""
            
            # Build product name for search
            product_name = f"{brand} {product_line} {shade}".strip()
            
            print(f"🔍 Processing {idx+1}/{len(df)}: {product_name}")
            
            # Build metadata filter
            meta = build_metadata_filter(
                sku=sku or None,
                category=category or None,
            )
            
            # Search for contexts
            contexts = search_contexts(
                query=product_name,
                namespace=namespace,
                metadata_filter=meta or None,
            )
            
            if not contexts:
                print(f"⚠️  No contexts found for: {product_name}")
                df.at[idx, 'attributes_json'] = json.dumps({"error": "no_contexts_found"})
                error_count += 1
                continue
            
            # Generate answer with Claude
            answer = generate_answer_with_claude(
                query=product_name,
                contexts=contexts,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Extract JSON attributes from response
            attributes = extract_json_from_response(answer)
            
            if attributes:
                # Convert to compact JSON string for Excel cell
                attributes_json = json.dumps(attributes, separators=(',', ':'), ensure_ascii=False)
                df.at[idx, 'attributes_json'] = attributes_json
                print(f"✅ Extracted attributes: {len(attributes)} keys")
                processed_count += 1
            else:
                print(f"⚠️  No valid JSON extracted from response")
                df.at[idx, 'attributes_json'] = json.dumps({"error": "invalid_json_response"})
                error_count += 1
                
        except Exception as e:
            print(f"❌ Error processing row {idx+1}: {e}")
            df.at[idx, 'attributes_json'] = json.dumps({"error": str(e)})
            error_count += 1
    
    # Save complete output to local Excel and Google Sheets
    local_success = False
    google_sheets_success = False
    output_path = "failed"
    
    try:
        output_path = save_to_excel_file(df, output_excel)
        print(f"💾 Local Excel saved: {output_path}")
        local_success = True
    except Exception as e:
        print(f"⚠️  Failed to save local Excel file: {e}")
        output_path = "failed"
    
    # Try Google Sheets upload if configured
    try:
        if GOOGLE_SHEETS_CREDENTIALS_PATH and os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH):
            google_sheets_success = save_to_google_sheets(df)
        else:
            print("ℹ️  Google Sheets not configured (run setup_google_sheets.py to enable)")
    except Exception as e:
        print(f"⚠️  Failed to update Google Sheets: {e}")
    
    # Summary
    total_with_attributes = len([row for _, row in df.iterrows() if pd.notna(row.get('attributes_json')) and str(row.get('attributes_json')).strip() and not str(row.get('attributes_json')).strip().startswith('{"error"')])
    
    print(f"\n📈 Processing Summary:")
    print(f"   ✅ Successfully processed this run: {processed_count}")
    print(f"   ❌ Errors this run: {error_count}")
    print(f"   📊 Total rows: {len(df)}")
    print(f"   📋 Total with valid attributes: {total_with_attributes}")
    print(f"   📁 Input: CSV ({csv_path})")
    print(f"   📁 Local Output: {'✅' if local_success else '❌'} ({output_path})")
    print(f"   📊 Google Sheets: {'✅' if google_sheets_success else '❌'} ({GOOGLE_SHEETS_NAME})")
    print(f"   📋 Both outputs contain product details + attributes_json column")
    
    if existing_df is not None and not force_reprocess:
        print(f"   🔄 Checkpoint: Loaded existing data and processed only missing attributes")
    elif force_reprocess:
        print(f"   🔄 Force Reprocess: All products were reprocessed regardless of existing attributes")
    
    return 0


def _cli_main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point - reads from CSV, saves to Excel."""
    return process_csv_to_excel_attributes(csv_path=DEFAULT_CSV_PATH)


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
