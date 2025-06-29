#!/usr/bin/env python3

"""
Debug script to understand why script categorization is failing
"""

import sys
import os
sys.path.append('core')
from nano_engine import NanoScriptEngine

def debug_categorization():
    engine = NanoScriptEngine()
    
    # Test the scripts that were used in our benchmark
    test_scripts = [
        'nano_scripts/small_script.py',
        'nano_scripts/medium_script.py', 
        'nano_scripts/large_script.py'
    ]
    
    print("=== Script Categorization Debug ===")
    print(f"Small threshold: {engine.SMALL_SCRIPT_THRESHOLD} lines")
    print(f"Medium threshold: {engine.MEDIUM_SCRIPT_THRESHOLD} lines")
    print()
    
    for script_path in test_scripts:
        print(f"Analyzing: {script_path}")
        
        # Check if file exists
        exists = os.path.exists(script_path)
        print(f"  File exists: {exists}")
        
        # Analyze complexity
        profile = engine.analyze_script_complexity(script_path)
        print(f"  Line count: {profile['line_count']}")
        print(f"  Complexity score: {profile['complexity_score']}")
        print(f"  Size category: {profile['size_category']}")
        
        # Show hash calculation for simulation
        hash_value = hash(script_path)
        simulated_lines = hash_value % 300
        print(f"  Hash value: {hash_value}")
        print(f"  Simulated lines (hash % 300): {simulated_lines}")
        
        # Manual categorization check
        manual_category = engine._categorize_script_size(profile['line_count'])
        print(f"  Manual categorization: {manual_category}")
        print()

if __name__ == "__main__":
    debug_categorization()
