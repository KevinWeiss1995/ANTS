import pandas as pd
import os

# Load the CSV file
data_path = "/Users/kweiss/git/cyber/ANTS/data/DDoS/all_data.csv"
data = pd.read_csv(data_path)

# Separate data into different CSV files based on the 'label' column
for label, group in data.groupby('Label'):
    output_file = os.path.join('/Users/kweiss/git/cyber/ANTS/data/DDoS/', f"{label}.csv")
    group.to_csv(output_file, index=False)
