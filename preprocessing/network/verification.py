import pandas as pd
scaled_data = pd.read_csv('data/network/scaled_data.csv')
print("Scaled data shape:", scaled_data.shape)
print("Unique traffic type values:", scaled_data["traffic type"].unique())

train_bin_labels = pd.read_csv('data/network/train_bin_trff_lbll_enc.csv', header=None)
print("Train binary labels sample:\n", train_bin_labels.head())
