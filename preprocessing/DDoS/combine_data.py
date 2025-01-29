import pandas as pd
import os

# Get the path relative to the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'DDoS'))

benign_df = pd.read_csv(os.path.join(data_dir, "BENIGN_sampled.csv"))
ddos_df = pd.read_csv(os.path.join(data_dir, "DDoS.csv"))

combined_df = pd.concat([benign_df, ddos_df]).sample(frac=1, random_state=42).reset_index(drop=True)
combined_df.to_csv(os.path.join(data_dir, "TRAINING_DATA.csv"), index=False)