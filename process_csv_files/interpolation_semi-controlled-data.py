import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
TARGET_DIR = Path("")
OUTPUT_DIR = Path("Interpolated")
# ---------------------

def process_file(filepath: Path, output_dir: Path):
    """
    Loads a CSV, interpolates specific columns, and saves to a new directory.
    Handles 'contact_points' with ffill as an exception.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        print(f"--- Processing {filepath.name} ---")
        
        # --- EXCEPTION LOGIC ---
        # 1. Handle the 'contact_points' exception first (ffill)
        # ffill() propagates the last valid observation forward.
        if 'contact_points' in df.columns:
            print("Found 'contact_points'. Applying forward fill (ffill).")
            df.pop('contact_points')
            a = 'contact_point' in df.columns.to_list()
            print(f"Contact_point still in df: {a}")
            #df['contact_points'] = pd.eval(df['contact_points'],)
            #df['contact_points'] = df['contact_points'].ffill()
        # -------------------------

        # --- STANDARD LOGIC ---
        # 2. Identify columns for linear interpolation
        # Get all columns matching the regex
        linear_cols_all = cols_to_interpolate = [col for col in df.columns 
                            if df[col].isna().any() or (df[col] == "").any()]
        
        # Exclude 'contact_points' from the list for linear interpolation
        cols_to_interpolate = [col for col in linear_cols_all if col != 'contact_points']

        if cols_to_interpolate:
            #print(f"Found columns to linearly interpolate: {cols_to_interpolate}")
            
            # 3. Apply linear interpolation to the remaining matches
            df[cols_to_interpolate] = df[cols_to_interpolate].apply(pd.to_numeric, 
                                errors="coerce").replace("",np.nan).interpolate(method='linear',limit_direction="both")
            
        else:
            print("No matching columns found for linear interpolation.")

        # 4. Define output path and save the file
        print(f"Number of null values in df: {df[cols_to_interpolate].isna().sum().sum()}")
        filename = 'interpolated_'+filepath.name
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Saved modified file to: {output_path}\n")

    except Exception as e:
        print(f"Error processing {filepath.name}: {e}\n")

def main():
    """
    Main function to run the batch processing.
    """
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nScanning for CSV files in: {TARGET_DIR.resolve()}")
    
    # Find all .csv files in the target directory
    csv_files = list(TARGET_DIR.glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in the target directory.")
        return

    # Process each file
    for csv_file in csv_files:
        process_file(csv_file, OUTPUT_DIR)
        
    print("--- All files processed. ---")

if __name__ == "__main__":
    main()