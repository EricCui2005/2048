import pandas as pd
import glob
import os
import tqdm

def combine_large_csv_files(folder_path, output_file, chunk_size=10000):
    # Get list of CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    
    for i, file in enumerate(csv_files):
        print(i)
        # Process each file in chunks
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            chunk.to_csv(output_file, mode='a', index=False)

# Example usage
folder_path = "combiningFolder"
output_file = "combined_large_output.csv"
combine_large_csv_files(folder_path, output_file)
