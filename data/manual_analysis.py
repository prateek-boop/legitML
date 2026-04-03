#!/usr/bin/env python3
"""
Manual analysis of archives and CSV files
"""

import os
import zipfile
import csv
import sys
from collections import defaultdict

def main():
    data_dir = r'C:\Users\prate\OneDrive\Desktop\legit0\data'
    
    print("=" * 60)
    print("ARCHIVE EXTRACTION AND CSV ANALYSIS")
    print("=" * 60)
    print(f"\nWorking Directory: {data_dir}\n")
    
    # Check what files exist
    print("=== CHECKING FOR ZIP FILES ===\n")
    zip_files_to_extract = [
        ('archive.zip', 'Archive 1'),
        ('archive(1).zip', 'Archive 2 (with space)'),
        ('top-1m.csv.zip', 'Top 1 Million CSV')
    ]
    
    for zip_name, description in zip_files_to_extract:
        zip_path = os.path.join(data_dir, zip_name)
        print(f"File: {zip_name}")
        print(f"Description: {description}")
        print(f"Exists: {os.path.exists(zip_path)}")
        
        if os.path.exists(zip_path):
            try:
                zip_size = os.path.getsize(zip_path)
                print(f"Size: {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
                
                # List contents
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        file_list = zf.namelist()
                        print(f"Contains {len(file_list)} file(s):")
                        for fname in file_list[:10]:  # Show first 10
                            print(f"  - {fname}")
                        if len(file_list) > 10:
                            print(f"  ... and {len(file_list) - 10} more")
                        
                        # Try to extract
                        print(f"\nExtracting to {data_dir}...")
                        zf.extractall(data_dir)
                        print("✓ Extraction successful\n")
                except zipfile.BadZipFile as e:
                    print(f"✗ Bad zip file: {e}\n")
                except Exception as e:
                    print(f"✗ Extraction error: {e}\n")
            except Exception as e:
                print(f"Error: {e}\n")
        else:
            print(f"✗ File does not exist\n")
        
        print("-" * 60 + "\n")
    
    # Now find all CSV files
    print("\n" + "=" * 60)
    print("=== FINDING AND ANALYZING CSV FILES ===")
    print("=" * 60 + "\n")
    
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir)
                csv_files.append((full_path, rel_path))
    
    csv_files.sort(key=lambda x: x[1])
    
    if not csv_files:
        print("No CSV files found!\n")
    else:
        print(f"Found {len(csv_files)} CSV file(s):\n")
        
        for full_path, rel_path in csv_files:
            print("=" * 60)
            print(f"File: {rel_path}")
            
            # Get file size
            try:
                file_size = os.path.getsize(full_path)
                size_mb = file_size / (1024 * 1024)
                size_gb = file_size / (1024 * 1024 * 1024)
                if size_gb > 1:
                    print(f"Size: {file_size:,} bytes ({size_gb:.2f} GB)")
                else:
                    print(f"Size: {file_size:,} bytes ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"Size: Error - {e}")
                print("-" * 60 + "\n")
                continue
            
            # Analyze CSV
            try:
                row_count = 0
                headers = None
                sample_rows = []
                
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    csv_reader = csv.reader(f)
                    
                    # Read header
                    headers = next(csv_reader, None)
                    
                    # Read all rows and collect samples
                    for i, row in enumerate(csv_reader):
                        row_count += 1
                        if i < 3:  # Store first 3 data rows as sample
                            sample_rows.append(row)
                
                if headers:
                    print(f"Columns ({len(headers)}): {', '.join(headers[:5])}", end="")
                    if len(headers) > 5:
                        print(f", ... and {len(headers) - 5} more columns")
                    else:
                        print()
                    
                    print(f"Data rows: {row_count:,}")
                    
                    # Show sample data
                    if sample_rows:
                        print("\nSample data (first 3 rows):")
                        for i, row in enumerate(sample_rows, 1):
                            print(f"  Row {i}: {row[:3]}...", end="")
                            if len(row) > 3:
                                print(f" ({len(row)} columns)")
                            else:
                                print()
                else:
                    print("Columns: (empty file)")
                    print("Data rows: 0")
            except Exception as e:
                print(f"Error analyzing CSV: {e}")
            
            print("-" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print("=== ANALYSIS COMPLETE ===")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
