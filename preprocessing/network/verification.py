import pandas as pd
import subprocess
import os


def get_git_repo_root():
    """Dynamically determine the Git repository root."""
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        print("Error: Not a git repository.")
        return None
    

base_repo = get_git_repo_root()

data_path = os.path.join(base_repo, "data", "network", "scaled_data.csv")
scaled_data = pd.read_csv(data_path, header=None)
train_bin_path = os.path.join(base_repo, 'data', 'network', 'train_bin_trff_lbll_enc.csv')
print("Scaled data shape:", scaled_data.shape)
print("Unique traffic type values:", scaled_data["traffic type"].unique())

train_bin_labels = pd.read_csv(train_bin_path, header=None)
print("Train binary labels sample:\n", train_bin_labels.head())
