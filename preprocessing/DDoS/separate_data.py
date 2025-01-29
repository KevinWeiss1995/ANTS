import pandas as pd
import os

# Get the path relative to the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'DDoS'))

# Load the CSV file
data_path = os.path.join(data_dir, "all_data.csv")
data = pd.read_csv(data_path)

# Separate data into different CSV files based on the 'label' column
for label, group in data.groupby('Label'):
    output_file = os.path.join(data_dir, f"{label}.csv")
    group.to_csv(output_file, index=False)
