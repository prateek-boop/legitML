#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.getcwd())

# Test if we can run
print("Python is working!")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Now run the analysis
import zipfile
import pandas as pd
from pathlib import Path

data_dir = Path(r"C:\Users\prate\OneDrive\Desktop\legit0\data")
print(f"\nData directory: {data_dir}")
print(f"Directory exists: {data_dir.exists()}")

# Extract all zips
print("\n" + "="*60)
print("EXTRACTING ZIP FILES:")
print("="*60)

for zip_file in sorted(data_dir.glob("*.zip")):
    print(f"\nExtracting: {zip_file.name}")
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            contents = z.namelist()
            print(f"  Contains {len(contents)} file(s): {contents}")
            z.extractall(data_dir)
            print(f"  ✓ Extracted successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("CSV FILES FOUND:")
print("="*60)

# Find and analyze all CSVs
csv_files = list(data_dir.glob("**/*.csv"))
print(f"\nTotal CSV files found: {len(csv_files)}")

for csv_file in sorted(csv_files):
    print(f"\n📄 {csv_file.name}")
    print(f"   Path: {csv_file}")
    try:
        # Read first 5 rows for preview
        df = pd.read_csv(csv_file, nrows=5, encoding='utf-8', on_bad_lines='skip')
        print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"   Preview shape: {df.shape}")
        
        # Get full row count
        with open(csv_file, encoding='utf-8', errors='ignore') as f:
            total = sum(1 for _ in f) - 1  # -1 for header
        print(f"   Total rows: {total:,}")
        print(f"   File size: {csv_file.stat().st_size:,} bytes")
    except Exception as e:
        print(f"   ✗ Error reading: {e}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
