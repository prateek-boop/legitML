#!/usr/bin/env python3
"""
Direct file processing and analysis
This script manually extracts and analyzes all archives and CSVs
"""

import os
import zipfile
import csv

os.chdir(r'C:\Users\prate\OneDrive\Desktop\legit0\data')
cwd = os.getcwd()

print("\n" + "="*80)
print("ARCHIVE EXTRACTION AND CSV FILE ANALYSIS")
print("="*80)
print(f"\nWorking Directory: {cwd}\n")

# STEP 1 & 2: Extract ZIP files
print("STEP 1 & 2: EXTRACTING ARCHIVES")
print("-"*80)

zip_files = {
    'archive.zip': 'Archive File 1',
    'archive(1).zip': 'Archive File 2',
    'top-1m.csv.zip': 'Top 1 Million CSV'
}

extraction_summary = []

for zip_name, description in zip_files.items():
    zip_path = os.path.join(cwd, zip_name)
    print(f"\n{zip_name}:")
    print(f"  Description: {description}")
    
    if not os.path.exists(zip_path):
        print(f"  Status: ✗ FILE NOT FOUND")
        extraction_summary.append(f"{zip_name}: NOT FOUND")
        continue
    
    file_size = os.path.getsize(zip_path)
    print(f"  File Size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            print(f"  Contains: {len(file_list)} file(s)")
            
            # List first few files
            for i, fname in enumerate(file_list[:5]):
                print(f"    - {fname}")
            if len(file_list) > 5:
                print(f"    - ... and {len(file_list) - 5} more files")
            
            # Extract
            print(f"  Extracting...", end=" ")
            zf.extractall(cwd)
            print("✓ SUCCESS")
            extraction_summary.append(f"{zip_name}: {len(file_list)} files extracted")
            
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        extraction_summary.append(f"{zip_name}: ERROR - {str(e)[:50]}")

# STEP 3: Find all CSV files
print("\n" + "="*80)
print("STEP 3: FINDING ALL CSV FILES")
print("-"*80)

csv_files_found = []
for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.lower().endswith('.csv'):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, cwd)
            csv_files_found.append((full_path, rel_path))

csv_files_found.sort(key=lambda x: x[1])

print(f"\nFound {len(csv_files_found)} CSV file(s):")
for i, (_, rel_path) in enumerate(csv_files_found, 1):
    print(f"  {i}. {rel_path}")

# STEP 4: Analyze each CSV file
print("\n" + "="*80)
print("STEP 4: CSV FILE ANALYSIS")
print("="*80)

analysis_results = []

for full_path, rel_path in csv_files_found:
    print(f"\n{'-'*80}")
    print(f"FILE: {rel_path}")
    print(f"{'-'*80}")
    
    try:
        # Get file size
        file_size = os.path.getsize(full_path)
        size_mb = file_size / (1024 * 1024)
        size_gb = file_size / (1024 * 1024 * 1024)
        
        if size_gb >= 1:
            size_str = f"{size_gb:.2f} GB"
        else:
            size_str = f"{size_mb:.2f} MB"
        
        print(f"File Size: {file_size:,} bytes ({size_str})")
        
        # Analyze CSV structure
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            
            if headers:
                print(f"Column Count: {len(headers)}")
                print(f"Column Names: {', '.join(headers)}")
                
                # Count data rows
                row_count = sum(1 for _ in reader)
                print(f"Data Rows: {row_count:,}")
                
                analysis_results.append({
                    'file': rel_path,
                    'size': file_size,
                    'columns': len(headers),
                    'rows': row_count,
                    'headers': headers
                })
            else:
                print("Status: Empty file (no headers)")
                analysis_results.append({
                    'file': rel_path,
                    'size': file_size,
                    'columns': 0,
                    'rows': 0,
                    'headers': []
                })
                
    except Exception as e:
        print(f"ERROR: {e}")
        analysis_results.append({
            'file': rel_path,
            'size': 0,
            'columns': 0,
            'rows': 0,
            'error': str(e)
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nArchives Processed: {len(extraction_summary)}")
for summary in extraction_summary:
    print(f"  - {summary}")

print(f"\nCSV Files Found: {len(csv_files_found)}")
print(f"CSV Files Analyzed: {len(analysis_results)}")

print("\nCSV Analysis Summary:")
for result in analysis_results:
    total_size_mb = result.get('size', 0) / (1024*1024)
    if 'error' in result:
        print(f"  {result['file']}: ERROR")
    else:
        print(f"  {result['file']}: {result['columns']} columns, {result['rows']:,} rows, {total_size_mb:.2f} MB")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")
