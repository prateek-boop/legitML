import os
import sys
import zipfile
import csv
import io
from pathlib import Path

# Add the data directory to Python path
data_dir = r'C:\Users\prate\OneDrive\Desktop\legit0\data'
os.chdir(data_dir)

print(f"Working directory: {data_dir}\n")

# Step 1 & 2: Extract all zip files
print("=== EXTRACTING ARCHIVES ===")
zip_files = ['archive.zip', 'archive(1).zip', 'top-1m.csv.zip']

for zip_file in zip_files:
    zip_path = os.path.join(data_dir, zip_file)
    if os.path.exists(zip_path):
        print(f"Extracting {zip_file}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"  Contains {len(file_list)} file(s):")
                # Show all files in the archive
                for fname in file_list:
                    print(f"    - {fname}")
                # Extract all
                zip_ref.extractall(data_dir)
            print(f"  ✓ Successfully extracted {zip_file}\n")
        except Exception as e:
            print(f"  ✗ Error extracting {zip_file}: {e}\n")
    else:
        print(f"  ✗ File not found: {zip_file}\n")

print("=== SCANNING FOR CSV FILES ===")
# Step 3: Find all CSV files recursively
csv_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

csv_files.sort()

if not csv_files:
    print("No CSV files found!")
else:
    print(f"Found {len(csv_files)} CSV file(s):\n")
    
    print("=== CSV FILE DETAILS ===\n")
    for csv_path in csv_files:
        rel_path = os.path.relpath(csv_path, data_dir)
        print(f"File: {rel_path}")
        
        # Get file size
        try:
            file_size = os.path.getsize(csv_path)
            print(f"Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"Size: Error - {e}")
            print()
            continue
        
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                
                # Get column names
                header = next(reader, None)
                if header:
                    print(f"Columns ({len(header)}): {', '.join(header)}")
                    
                    # Count data rows
                    row_count = sum(1 for _ in reader)
                    print(f"Data rows: {row_count:,}")
                else:
                    print("Columns: (empty file)")
                    print("Data rows: 0")
        except Exception as e:
            print(f"Error reading file: {e}")
        
        print()

print("=== COMPLETE ===")
