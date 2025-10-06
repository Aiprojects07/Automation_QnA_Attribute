#!/usr/bin/env python3
"""
Google Sheets Setup Helper
This script helps you set up Google Sheets integration for automatic uploading.
"""

import os
import json

def setup_google_sheets():
    """Interactive setup for Google Sheets integration."""
    
    print("🔧 Google Sheets Setup Helper")
    print("=" * 50)
    
    # Check if credentials file exists
    creds_path = input("📁 Enter path to your Google service account JSON file: ").strip()
    
    if not creds_path:
        print("\n📝 To get the JSON credentials file:")
        print("1. Go to: https://console.cloud.google.com/")
        print("2. Create a new project or select existing")
        print("3. Enable Google Sheets API")
        print("4. Create Service Account:")
        print("   - Go to APIs & Services → Credentials")
        print("   - Create Credentials → Service Account")
        print("   - Download JSON key file")
        print("5. Share your Google Sheet with the service account email")
        return False
    
    if not os.path.exists(creds_path):
        print(f"❌ File not found: {creds_path}")
        return False
    
    # Validate JSON file
    try:
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
        
        service_email = creds_data.get('client_email', 'Unknown')
        project_id = creds_data.get('project_id', 'Unknown')
        
        print(f"✅ Valid credentials file found!")
        print(f"   Service Account: {service_email}")
        print(f"   Project ID: {project_id}")
        
    except Exception as e:
        print(f"❌ Invalid JSON file: {e}")
        return False
    
    # Get sheet details
    sheet_name = input("\n📊 Enter your Google Sheet name: ").strip()
    if not sheet_name:
        sheet_name = "Lipstick Attributes"
    
    worksheet_name = input("📋 Enter worksheet name (default: Sheet1): ").strip()
    if not worksheet_name:
        worksheet_name = "Sheet1"
    
    # Create .env file
    env_path = "/home/sid/Documents/Automation_QnA_Attribute/.env"
    
    env_content = f"""# Google Sheets Configuration
GOOGLE_SHEETS_CREDENTIALS_PATH={creds_path}
GOOGLE_SHEETS_NAME={sheet_name}
GOOGLE_SHEETS_WORKSHEET={worksheet_name}

# Add your other API keys here
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# PINECONE_API_KEY=your_key_here
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Configuration saved to: {env_path}")
    print("\n📋 Next steps:")
    print("1. Make sure your Google Sheet is shared with:")
    print(f"   {service_email}")
    print("2. Give the service account 'Editor' permissions")
    print("3. Run your attribute extraction script")
    print("\n🎉 Google Sheets integration is ready!")
    
    return True

def test_google_sheets():
    """Test Google Sheets connection."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        import sys
        sys.path.insert(0, '/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/core')
        from search_and_answer import save_to_google_sheets
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'Product': ['Test Lipstick'],
            'Brand': ['Test Brand'],
            'Attributes': ['{"finish": "matte"}']
        })
        
        print("🧪 Testing Google Sheets connection...")
        success = save_to_google_sheets(test_data)
        
        if success:
            print("✅ Google Sheets test successful!")
        else:
            print("❌ Google Sheets test failed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Setup Google Sheets integration")
    print("2. Test Google Sheets connection")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        setup_google_sheets()
    elif choice == "2":
        test_google_sheets()
    else:
        print("Invalid choice")
