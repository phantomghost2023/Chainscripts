"""
Nano-script: CSV Cleaner
Atomic script for cleaning CSV data
"""

import pandas as pd
import sys
import argparse
from pathlib import Path

def clean_csv(input_file: str, output_file: str = None, remove_duplicates: bool = True, 
              fill_na: str = None, standardize_headers: bool = True):
    """
    Clean CSV data with various options
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        remove_duplicates: Remove duplicate rows
        fill_na: Value to fill NaN values with
        standardize_headers: Convert headers to lowercase and replace spaces with underscores
    """
    try:
        # Read CSV
        df = pd.read_csv(input_file)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Standardize headers
        if standardize_headers:
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            print("Standardized column headers")
        
        # Remove duplicates
        if remove_duplicates:
            before_count = len(df)
            df = df.drop_duplicates()
            removed = before_count - len(df)
            if removed > 0:
                print(f"Removed {removed} duplicate rows")
        
        # Fill NaN values
        if fill_na is not None:
            df = df.fillna(fill_na)
            print(f"Filled NaN values with '{fill_na}'")
        
        # Save result
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Error cleaning CSV: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Clean CSV data")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path")
    parser.add_argument("--no-duplicates", action="store_false", dest="remove_duplicates",
                       help="Don't remove duplicates")
    parser.add_argument("--fill-na", help="Value to fill NaN with")
    parser.add_argument("--no-standardize", action="store_false", dest="standardize_headers",
                       help="Don't standardize headers")
    
    args = parser.parse_args()
    
    clean_csv(
        input_file=args.input,
        output_file=args.output,
        remove_duplicates=args.remove_duplicates,
        fill_na=args.fill_na,
        standardize_headers=args.standardize_headers
    )

if __name__ == "__main__":
    main()
