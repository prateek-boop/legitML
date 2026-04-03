import zipfile
import os
import pandas as pd
from pathlib import Path

data_dir = Path(r"C:\Users\prate\OneDrive\Desktop\legit0\data")

# Extract all zips
for zip_file in data_dir.glob("*.zip"):
    print(f"Extracting: {zip_file.name}")
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(data_dir)
            print(f"  Contents: {z.namelist()}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*60)
print("CSV FILES FOUND:")
print("="*60)

# Find and analyze all CSVs
for csv_file in data_dir.glob("**/*.csv"):
    print(f"\n📄 {csv_file.name}")
    try:
        df = pd.read_csv(csv_file, nrows=5, encoding='utf-8', on_bad_lines='skip')
        print(f"   Columns: {list(df.columns)}")
        print(f"   Shape preview: {df.shape}")
        
        # Get full row count
        total = sum(1 for _ in open(csv_file, encoding='utf-8', errors='ignore')) - 1
        print(f"   Total rows: {total:,}")
    except Exception as e:
        print(f"   Error reading: {e}")
