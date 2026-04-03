#!/usr/bin/env python3
if __name__ == '__main__':
    import os, sys, zipfile, csv
    os.chdir(r'C:\Users\prate\OneDrive\Desktop\legit0\data')
    print("\n" + "="*100)
    print("ARCHIVE EXTRACTION AND CSV FILE ANALYSIS")
    print("="*100 + f"\nWorking Directory: {os.getcwd()}\n")
    print("STEP 1-2: EXTRACTING ARCHIVES\n" + "-"*100)
    for zip_name in ['archive.zip', 'archive(1).zip', 'top-1m.csv.zip']:
        zip_path = os.path.join(os.getcwd(), zip_name)
        if not os.path.exists(zip_path):
            print(f"\n{zip_name}: NOT FOUND")
            continue
        print(f"\n{zip_name}:")
        print(f"  Size: {os.path.getsize(zip_path):,} bytes ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                files = zf.namelist()
                print(f"  Contains {len(files)} file(s):")
                for f in files: print(f"    - {f}")
                zf.extractall()
                print(f"  ✓ Extracted successfully")
        except Exception as e: print(f"  ✗ Error: {e}")
    print("\n" + "="*100)
    print("STEP 3: SCANNING FOR CSV FILES")
    print("-"*100)
    csv_files = []
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append((os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.getcwd())))
    csv_files.sort(key=lambda x: x[1])
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, (_, rel) in enumerate(csv_files, 1): print(f"  {i}. {rel}")
    print("\n" + "="*100)
    print("STEP 4: CSV FILE ANALYSIS")
    print("="*100)
    for full_path, rel_path in csv_files:
        print(f"\nFILE: {rel_path}")
        print("-"*100)
        try:
            size = os.path.getsize(full_path)
            print(f"Size: {size:,} bytes ({size/(1024*1024):.2f} MB)")
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    print(f"Columns ({len(headers)}): {', '.join(headers)}")
                    row_count = sum(1 for _ in reader)
                    print(f"Data rows: {row_count:,}")
                else: print("Empty file")
        except Exception as e: print(f"Error: {e}")
    print("\n" + "="*100)
    print("COMPLETE")
    print("="*100 + "\n")
