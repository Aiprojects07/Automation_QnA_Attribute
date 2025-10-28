#!/usr/bin/env python3
"""
Google Drive Upload Module
Reusable module for uploading files to Google Drive with OAuth authentication.
"""

import os
import pickle
import logging
from typing import Optional
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Scopes: only access files created by this app
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Default paths (relative to this file's location)
BASE_DIR = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.pickle"

# Target Google Drive folder ID for QnA JSON uploads
# ----GPt Output File Path # Folder: https://drive.google.com/drive/folders/1_dJUqPIptoH6Yfbr6VXI0urZzlO6_QOC?usp=sharing
TARGET_FOLDER_ID = "1_dJUqPIptoH6Yfbr6VXI0urZzlO6_QOC"


def get_drive_service():
    """Authenticate and return Google Drive service instance.
    
    On first run, opens browser for OAuth consent.
    Subsequent runs use cached token from token.pickle.
    
    Returns:
        Google Drive API service instance
    """
    creds = None
    
    # Load cached token if it exists
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            logging.warning(f"Failed to load cached token: {e}")
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logging.warning(f"Token refresh failed: {e}. Re-authenticating...")
                creds = None
        
        if not creds:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Credentials file not found: {CREDENTIALS_FILE}\n"
                    "Please download credentials.json from Google Cloud Console."
                )
            
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save token for future use
        try:
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        except Exception as e:
            logging.warning(f"Failed to save token: {e}")
    
    service = build('drive', 'v3', credentials=creds)
    return service


def find_file_in_folder(service, file_name: str, folder_id: str) -> Optional[str]:
    """Find a file by name in a specific Google Drive folder.
    
    Args:
        service: Google Drive API service instance
        file_name: Name of the file to search for
        folder_id: ID of the folder to search in
    
    Returns:
        File ID if found, None otherwise
    """
    try:
        # Search for file with exact name in the specified folder
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = results.get('files', [])
        if files:
            return files[0]['id']  # Return first match
        
        return None
    
    except HttpError as error:
        logging.error(f"Failed to search for file '{file_name}': {error}")
        return None


def find_or_create_folder(service, folder_name: str, parent_id: Optional[str] = None) -> str:
    """Find or create a folder in Google Drive.
    
    Args:
        service: Google Drive API service instance
        folder_name: Name of the folder
        parent_id: Optional parent folder ID (None = root)
    
    Returns:
        Folder ID
    """
    try:
        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = results.get('files', [])
        if files:
            return files[0]['id']
        
        # Create folder if not found
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]
        
        folder = service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()
        
        return folder.get('id')
    
    except HttpError as error:
        logging.error(f"Failed to find/create folder '{folder_name}': {error}")
        raise


def upload_file_to_drive(
    file_path: str,
    folder_name: Optional[str] = None,
    folder_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    replace_existing: bool = True
) -> bool:
    """Upload a file to Google Drive, optionally replacing existing files with same name.
    
    Args:
        file_path: Absolute path to the file to upload
        folder_name: Optional folder name in Drive (creates if doesn't exist)
        folder_id: Optional folder ID to upload to (takes precedence over folder_name)
        logger: Optional logger instance for logging
        replace_existing: If True, replaces existing file with same name (default: True)
    
    Returns:
        True if upload succeeded, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Get Drive service
        service = get_drive_service()
        
        # Get or create folder if specified
        parent_id = None
        if folder_id:
            # Use provided folder ID directly
            parent_id = folder_id
            logger.debug(f"Using Drive folder ID: {parent_id}")
        elif folder_name:
            # Create or find folder by name
            parent_id = find_or_create_folder(service, folder_name)
            logger.debug(f"Using Drive folder: {folder_name} (ID: {parent_id})")
        
        # Prepare file metadata
        file_name = os.path.basename(file_path)
        media = MediaFileUpload(file_path, resumable=True)
        
        # Check if file already exists in the folder
        existing_file_id = None
        if replace_existing and parent_id:
            existing_file_id = find_file_in_folder(service, file_name, parent_id)
        
        if existing_file_id:
            # Update existing file (replace content)
            logger.info(f"📄 File exists in Drive: {file_name} (ID: {existing_file_id})")
            logger.info(f"🔄 Replacing existing file with updated content...")
            file = service.files().update(
                fileId=existing_file_id,
                media_body=media,
                fields='id, webViewLink'
            ).execute()
            
            file_id = file.get('id')
            web_link = file.get('webViewLink')
            
            logger.info(f"✅ Replaced in Drive: {file_name} (ID: {file_id})")
            logger.debug(f"   View at: {web_link}")
        else:
            # Create new file
            file_metadata = {'name': file_name}
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()
            
            file_id = file.get('id')
            web_link = file.get('webViewLink')
            
            logger.info(f"✅ Uploaded to Drive: {file_name} (ID: {file_id})")
            logger.debug(f"   View at: {web_link}")
        
        return True
    
    except HttpError as error:
        logger.error(f"Google Drive API error for {file_path}: {error}")
        return False
    except Exception as error:
        logger.error(f"Failed to upload {file_path} to Drive: {error}")
        return False


def upload_multiple_files(
    file_paths: list[str],
    folder_name: Optional[str] = None,
    folder_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> tuple[int, int]:
    """Upload multiple files to Google Drive.
    
    Args:
        file_paths: List of absolute file paths to upload
        folder_name: Optional folder name in Drive
        folder_id: Optional folder ID to upload to (takes precedence)
        logger: Optional logger instance
    
    Returns:
        Tuple of (success_count, fail_count)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    success = 0
    failed = 0
    
    for file_path in file_paths:
        if upload_file_to_drive(file_path, folder_name, folder_id, logger):
            success += 1
        else:
            failed += 1
    
    logger.info(f"📤 Drive upload summary: {success} succeeded, {failed} failed")
    return success, failed


if __name__ == "__main__":
    # Test the module with actual QnA JSON file
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test file path
    test_file_path = "/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/output/vz8o6qmbccdq0ult266et-colorbar-velvet-matte-lipstick-wanna-be.json"
    
    try:
        # Test authentication
        service = get_drive_service()
        logger.info("✅ Authentication successful!")
        
        # Verify test file exists
        if not os.path.exists(test_file_path):
            logger.error(f"❌ Test file not found: {test_file_path}")
            exit(1)
        
        logger.info(f"📄 Test file: {os.path.basename(test_file_path)}")
        
        # Test upload to your specified folder
        logger.info(f"📤 Uploading to Drive folder: {TARGET_FOLDER_ID}...")
        success = upload_file_to_drive(
            file_path=test_file_path,
            folder_id=TARGET_FOLDER_ID,
            logger=logger
        )
        
        if success:
            logger.info("✅ TEST PASSED: File uploaded successfully!")
            logger.info(f"   Check your Drive folder: https://drive.google.com/drive/folders/{TARGET_FOLDER_ID}")
        else:
            logger.error("❌ TEST FAILED: Upload failed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
