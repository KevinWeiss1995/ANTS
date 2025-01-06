import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn as sk
import time

# Start timer
start_time = time.time()

# Function to create folders if they don't exist
def create_folder(folder_name):
    try:
        os.makedirs(folder_name, exist_ok=True)
    except OSError as e:
        print(f"Error creating folder {folder_name}: {e}")

# Define data directory and file paths
data_dir = "/Users/kweiss/git/ANTS/data"
csv_files = [os.path.join(data_dir, "all_data.csv")]  # List of dataset file paths

# Define main labels (features and target)
main_labels = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
]

# Output files and directories
output_dir = os.path.join(data_dir, "feature_pics")
output_csv = os.path.join(data_dir, "importance_list_all_data.csv")
create_folder(output_dir)

# Open output file for writing
with open(output_csv, "w") as ths:
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, usecols=main_labels)
        df.fillna(0, inplace=True)

        # Encode "BENIGN" as 1, others as 0
        df["Label"] = df["Label"].apply(lambda x: 1 if x == "BENIGN" else 0)

        y = df.pop("Label").values
        X = df.values

        # Train Random Forest model
        forest = RandomForestRegressor(n_estimators=250, random_state=0)
        forest.fit(X, y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Extract and save top features
        refclasscol = df.columns
        top_features = pd.DataFrame({
            'Features': refclasscol[indices[:20]],
            'Importance': importances[indices[:20]]
        }).set_index('Features').sort_values('Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 5))
        top_features.plot.bar()
        plt.title(f"{os.path.basename(csv_file)[:-4]} - Feature Importance")
        plt.ylabel('Importance')
        plot_path = os.path.join(output_dir, f"{os.path.basename(csv_file)[:-4]}.pdf")
        plt.savefig(plot_path, bbox_inches='tight', format='pdf')
        plt.close()

        # Write top 5 features to output file
        top_5_features = ", ".join([f'"{feat}"' for feat in top_features.index[:5]])
        ths.write(f"{os.path.basename(csv_file)[:-4]}=[{top_5_features}]\n")

        print(f"Processed {csv_file}: top features saved to {plot_path}")

# Print operation time
print(f"Mission accomplished! Total operation time: {time.time() - start_time:.2f} seconds")
