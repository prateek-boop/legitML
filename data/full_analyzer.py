#!/usr/bin/env python3
"""
Archive extraction and CSV analysis script
This script will be analyzed to determine what needs to be executed
"""

import os
import sys
import zipfile
import csv
from pathlib import Path

def extract_zip_files(directory):
    """Extract all zip files in the given directory"""
    print("=== EXTRACTING ARCHIVES ===")
    zip_files = [
        os.path.join(directory, 'archive.zip'),
        os.path.join(directory, 'archive(1).zip'),
        os.path.join(directory, 'top-1m.csv.zip')
    ]
    
    for zip_path in zip_files:
        if os.path.exists(zip_path):
            basename = os.path.basename(zip_path)
            print(f"Extracting {basename}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    zip_ref.extractall(directory)
                    print(f"  ✓ Successfully extracted {basename}")
                    print(f"    Files extracted: {len(file_list)}")
                    if file_list:
                        for fname in file_list[:5]:  # Show first 5 files
                            print(f"      - {fname}")
                        if len(file_list) > 5:
                            print(f"      ... and {len(file_list) - 5} more files")
            except Exception as e:
                print(f"  ✗ Error extracting {basename}: {e}")
        else:
            print(f"  ✗ File not found: {os.path.basename(zip_path)}")

def find_and_analyze_csv_files(directory):
    """Find and analyze all CSV files in the directory"""
    print("\n=== FINDING CSV FILES ===")
    
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    csv_files.sort()
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)\n")
    
    print("=== CSV FILE DETAILS ===\n")
    
    for csv_path in csv_files:
        rel_path = os.path.relpath(csv_path, directory)
        print(f"File: {rel_path}")
        
        # Get file size
        try:
            file_size = os.path.getsize(csv_path)
            size_mb = file_size / (1024 * 1024)
            print(f"Size: {file_size:,} bytes ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"Size: Error - {e}")
            continue
        
        # Analyze CSV
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

def main():
    # Set the working directory
    data_dir = r'C:\Users\prate\OneDrive\Desktop\legit0\data'
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return 1
    
    print(f"Working directory: {data_dir}\n")
    
    # Change to the data directory
    os.chdir(data_dir)
    
    # Extract zip files
    extract_zip_files(data_dir)
    
    # Find and analyze CSV files
    find_and_analyze_csv_files(data_dir)
    
    print("=== COMPLETE ===")
    return 0

if __name__ == '__main__':
    sys.exit(main())
