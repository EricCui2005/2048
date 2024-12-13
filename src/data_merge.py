import pandas as pd
import glob

# Create an empty DataFrame with the desired header
combined_df = pd.DataFrame(columns=['action', 'state'])

# Get all CSV files in the current directory
csv_files = glob.glob('data/*.csv')

# Combine all CSV files
for file in csv_files:
    df = pd.read_csv(file)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_data.csv', index=False)