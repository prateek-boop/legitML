#!/usr/bin/env python3
"""
Data Analysis Script - Extract zips and analyze CSV files
Run this with: python extract_and_analyze.py
"""

import zipfile
import os
import pandas as pd
from pathlib import Path
import sys

data_dir = Path(r"C:\Users\prate\OneDrive\Desktop\legit0\data")

def main():
    print("\n" + "="*70)
    print("DATA EXTRACTION AND ANALYSIS SCRIPT")
    print("="*70)
    
    # Extract all zips
    print("\n[1/3] EXTRACTING ZIP FILES...")
    print("-" * 70)
    
    zip_count = 0
    for zip_file in sorted(data_dir.glob("*.zip")):
        zip_count += 1
        print(f"\n  Extracting: {zip_file.name}")
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                contents = z.namelist()
                print(f"    Contains: {contents}")
                z.extractall(data_dir)
                print(f"    ✓ Successfully extracted")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    if zip_count == 0:
        print("  No zip files found!")
    
    # Find and analyze all CSVs
    print("\n[2/3] FINDING CSV FILES...")
    print("-" * 70)
    
    csv_files = sorted(data_dir.glob("**/*.csv"))
    print(f"\nTotal CSV files found: {len(csv_files)}")
    
    print("\n[3/3] ANALYZING CSV FILES...")
    print("-" * 70)
    
    for csv_file in csv_files:
        print(f"\n📄 File: {csv_file.name}")
        print(f"   Full path: {csv_file}")
        try:
            # Get file size
            file_size = csv_file.stat().st_size
            size_mb = file_size / (1024 * 1024)
            print(f"   File size: {file_size:,} bytes ({size_mb:.2f} MB)")
            
            # Read first 5 rows for preview
            df = pd.read_csv(csv_file, nrows=5, encoding='utf-8', on_bad_lines='skip')
            num_cols = len(df.columns)
            print(f"   Columns ({num_cols}): {list(df.columns)}")
            
            # Get full row count
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                total_rows = sum(1 for _ in f) - 1  # -1 for header
            
            print(f"   Total rows: {total_rows:,}")
            print(f"   First 5 rows preview:")
            print(f"   {df.to_string()[:300]}...")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)
