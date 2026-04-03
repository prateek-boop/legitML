import os
import sys
import zipfile
import csv

def test_extract_and_analyze():
    """Test function that extracts archives and analyzes CSV files"""
    
    os.chdir(r'C:\Users\prate\OneDrive\Desktop\legit0\data')
    cwd = os.getcwd()
    
    print("\n" + "="*100)
    print("ARCHIVE EXTRACTION AND CSV FILE ANALYSIS")
    print("="*100)
    print(f"\nWorking Directory: {cwd}\n")
    
    # STEP 1 & 2: Extract ZIP files
    print("STEP 1-2: EXTRACTING ARCHIVES")
    print("-"*100)
    
    zip_files = {
        'archive.zip': 'Archive File 1',
        'archive(1).zip': 'Archive File 2 (note: has spaces in filename)',
        'top-1m.csv.zip': 'Top 1 Million Websites CSV'
    }
    
    extraction_results = {}
    
    for zip_name, description in zip_files.items():
        zip_path = os.path.join(cwd, zip_name)
        print(f"\nFile: {zip_name}")
        print(f"Description: {description}")
        
        if not os.path.exists(zip_path):
            print(f"Status: NOT FOUND")
            extraction_results[zip_name] = "NOT FOUND"
            continue
        
        file_size = os.path.getsize(zip_path)
        print(f"Size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = zf.namelist()
                print(f"Archive contains {len(file_list)} file(s):")
                
                # List all files
                for fname in file_list:
                    print(f"  - {fname}")
                
                # Extract
                print(f"\nExtracting...", end=" ")
                sys.stdout.flush()
                zf.extractall(cwd)
                print("✓ SUCCESS")
                extraction_results[zip_name] = f"{len(file_list)} files extracted"
                
        except Exception as e:
            print(f"ERROR: {e}")
            extraction_results[zip_name] = f"ERROR: {str(e)[:80]}"
    
    # STEP 3: Find all CSV files
    print("\n" + "="*100)
    print("STEP 3: SCANNING FOR CSV FILES")
    print("-"*100)
    
    csv_files_found = []
    for root, dirs, files in os.walk(cwd):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, cwd)
                csv_files_found.append((full_path, rel_path))
    
    csv_files_found.sort(key=lambda x: x[1])
    
    print(f"\nCSV Files Found: {len(csv_files_found)}\n")
    for i, (_, rel_path) in enumerate(csv_files_found, 1):
        print(f"  {i}. {rel_path}")
    
    # STEP 4: Analyze each CSV file
    print("\n" + "="*100)
    print("STEP 4: DETAILED CSV FILE ANALYSIS")
    print("="*100)
    
    for full_path, rel_path in csv_files_found:
        print(f"\n{'='*100}")
        print(f"FILE: {rel_path}")
        print(f"{'='*100}")
        
        try:
            # Get file size
            file_size = os.path.getsize(full_path)
            size_mb = file_size / (1024 * 1024)
            size_gb = file_size / (1024 * 1024 * 1024)
            
            if size_gb >= 1:
                size_display = f"{size_gb:.2f} GB"
            else:
                size_display = f"{size_mb:.2f} MB"
            
            print(f"File Size: {file_size:,} bytes ({size_display})")
            
            # Analyze CSV
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                
                if headers:
                    print(f"Column Count: {len(headers)}")
                    print(f"\nColumn Names ({len(headers)} total):")
                    for col_num, col_name in enumerate(headers, 1):
                        print(f"  {col_num:2d}. {col_name}")
                    
                    # Count data rows
                    print(f"\nCounting data rows...", end=" ")
                    sys.stdout.flush()
                    row_count = sum(1 for _ in reader)
                    print(f"Done!")
                    print(f"Data Rows: {row_count:,}")
                    print(f"Total Rows (including header): {row_count + 1:,}")
                else:
                    print("Status: Empty file (no header row found)")
                    
        except Exception as e:
            print(f"ERROR analyzing file: {e}")
    
    # Final summary
    print("\n" + "="*100)
    print("EXTRACTION AND ANALYSIS COMPLETE")
    print("="*100)
    
    print(f"\nArchive Extraction Summary:")
    for zip_name, result in extraction_results.items():
        print(f"  {zip_name}: {result}")
    
    print(f"\nCSV Files Processed: {len(csv_files_found)}")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    test_extract_and_analyze()
