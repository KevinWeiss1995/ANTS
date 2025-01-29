import pandas as pd
import os

# Get the path relative to the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'DDoS'))

# Load and sample the data
df = pd.read_csv(os.path.join(data_dir, "BENIGN.csv"))
sampled_df = df.sample(n=83670, random_state=42)
sampled_df.to_csv(os.path.join(data_dir, "BENIGN_sampled.csv"), index=False)


