#!/usr/bin/env python3
import subprocess
import sys

result = subprocess.run([
    sys.executable, 
    r'C:\Users\prate\OneDrive\Desktop\legit0\data\manual_analysis.py'
], capture_output=False, text=True)

sys.exit(result.returncode)
