import pandas as pd

df = pd.read_csv("/Users/kweiss/git/cyber/ANTS/data/DDoS/BENIGN.csv")
sampled_df = df.sample(n=83670, random_state=42)
sampled_df.to_csv("/Users/kweiss/git/cyber/ANTS/data/DDoS/BENIGN_sampled.csv", index=False)


