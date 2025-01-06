import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import time

seconds = time.time()

# Function to create a folder if it doesn't exist
def folder(f_name):
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print("The folder could not be created!")

# List CSV files in the "attacks" folder
csv_files = os.listdir("attacks")

# Define the main columns for the dataset
main_labels = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
]

# Create output folder for feature importance plots
folder("./feature_pics/")

# Open file for saving the feature importance results
with open("importance_list_for_attack_files.csv", "w") as ths:
    for j in csv_files:
        # Read the dataset
        df = pd.read_csv("./attacks/" + j, usecols=main_labels)
        df = df.fillna(0)  # Fill missing values with 0

        # Convert "BENIGN" to 1 and other labels to 0
        attack_or_not = [1 if i == "BENIGN" else 0 for i in df["Label"]]
        df["Label"] = attack_or_not

        # Prepare features and labels
        y = df["Label"].values
        del df["Label"]
        X = df.values

        # Train the RandomForest model
        forest = RandomForestRegressor(n_estimators=250, random_state=0)
        forest.fit(X, y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Prepare feature importance data
        refclasscol = list(df.columns.values)
        impor_bars = pd.DataFrame({'Features': refclasscol[0:20], 'importance': importances[0:20]})
        impor_bars = impor_bars.sort_values('importance', ascending=False).set_index('Features')

        # Plot feature importance
        plt.rcParams['figure.figsize'] = (10, 5)
        impor_bars.plot.bar()
        plt.title(j[0:-4] + " Attack - Feature Importance")
        plt.ylabel('Importance')
        plt.tight_layout()

        # Save the plot as a PDF
        plt.savefig("./feature_pics/" + j[0:-4] + ".pdf", bbox_inches='tight', format='pdf')
        plt.show()

        # Prepare and write feature importance data to file
        count = 0
        fea_ture = j[0:-4] + "=["
        for i in impor_bars.index:
            fea_ture += "\"" + str(i) + "\","  # Add feature to the list
            count += 1
            if count == 5:  # Limit to top 5 features
                fea_ture = fea_ture[0:-1] + "]"  # Remove the trailing comma
                break

        print(j[0:-4], "importance list:")
        print(j[0:-4], "\n", impor_bars.head(20), "\n\n\n")
        print(fea_ture)

        # Write the feature importance data to the file
        ths.write(fea_ture + "\n")

    print("Mission accomplished!")
    print("Total operation time: = ", time.time() - seconds, "seconds")
