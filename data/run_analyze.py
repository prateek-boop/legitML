#!/usr/bin/env python
import subprocess
import sys

# Change to the script directory and run analyze.py
result = subprocess.run([sys.executable, 'analyze.py'], cwd=r'C:\Users\prate\OneDrive\Desktop\legit0\data', capture_output=False, text=True)
sys.exit(result.returncode)
