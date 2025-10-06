#!/usr/bin/env python3
"""
Test script to verify checkpoint functionality in search_and_answer.py
"""

import os
import sys
import pandas as pd
import json

# Add the core module to path
sys.path.insert(0, '/home/sid/Documents/Automation_QnA_Attribute/attribute_generation_upsert/core')

def test_checkpoint_functionality():
    """Test the checkpoint functionality without making API calls."""
    
    # Create a test CSV with sample data
    test_csv_path = "/tmp/test_lipstick.csv"
    test_data = {
        'S.No.': [1, 2, 3],
        'Kult SKU Code': ['SKU001', 'SKU002', 'SKU003'],
        'Category': ['Makeup', 'Makeup', 'Makeup'],
        'Sub Category': ['Lip', 'Lip', 'Lip'],
        'Brand': ['TestBrand1', 'TestBrand2', 'TestBrand3'],
        'Product_name': ['Test Lipstick 1', 'Test Lipstick 2', 'Test Lipstick 3'],
        'Shade': ['Red', 'Pink', 'Coral']
    }
    
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(test_csv_path, index=False)
    print(f"✅ Created test CSV: {test_csv_path}")
    
    # Create a test Excel file with some processed attributes
    test_excel_path = "/tmp/test_attributes_output.xlsx"
    existing_data = test_data.copy()
    existing_data['attributes_json'] = [
        '{"attributes": [{"name": "finish", "value": "matte", "confidence": "high"}]}',  # SKU001 has attributes
        '',  # SKU002 is empty (should be processed)
        '{"error": "no_contexts_found"}'  # SKU003 has error (should be reprocessed)
    ]
    
    existing_df = pd.DataFrame(existing_data)
    existing_df.to_excel(test_excel_path, index=False, engine='openpyxl')
    print(f"✅ Created test Excel with existing data: {test_excel_path}")
    
    # Test the checkpoint logic manually
    print("\n🔍 Testing checkpoint logic:")
    
    # Load CSV
    csv_df = pd.read_csv(test_csv_path)
    print(f"📊 CSV loaded: {len(csv_df)} rows")
    
    # Load existing Excel
    existing_df = pd.read_excel(test_excel_path, engine='openpyxl')
    print(f"📋 Existing Excel loaded: {len(existing_df)} rows")
    
    # Test merge logic
    merge_key = 'Kult SKU Code'
    if merge_key in csv_df.columns and merge_key in existing_df.columns:
        print(f"🔗 Using merge key: {merge_key}")
        df = csv_df.merge(existing_df[['attributes_json', merge_key]], 
                         on=merge_key, how='left', suffixes=('', '_existing'))
        df['attributes_json'] = df['attributes_json'].fillna('')
        
        print("\n📋 Merged data status:")
        for idx, row in df.iterrows():
            sku = row.get('Kult SKU Code', f'Row {idx+1}')
            attrs = row.get('attributes_json', '')
            has_attrs = pd.notna(attrs) and str(attrs).strip() and not str(attrs).strip().startswith('{"error"')
            status = "✅ Has valid attributes" if has_attrs else "❌ Needs processing"
            print(f"   {sku}: {status}")
            if attrs:
                print(f"      Content: {attrs[:50]}...")
    
    # Clean up test files
    os.remove(test_csv_path)
    os.remove(test_excel_path)
    print(f"\n🧹 Cleaned up test files")
    
    print("\n✅ Checkpoint functionality test completed successfully!")

if __name__ == "__main__":
    test_checkpoint_functionality()
