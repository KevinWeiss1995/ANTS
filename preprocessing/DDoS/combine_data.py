import pandas as pd

benign_df = pd.read_csv("/Users/kweiss/git/cyber/ANTS/data/DDoS/BENIGN_sampled.csv")
ddos_df = pd.read_csv("/Users/kweiss/git/cyber/ANTS/data/DDoS/DDoS.csv")

combined_df = pd.concat([benign_df, ddos_df]).sample(frac=1, random_state=42).reset_index(drop=True)
combined_df.to_csv("/Users/kweiss/git/cyber/ANTS/data/DDoS/TRAINING_DATA.csv", index=False)