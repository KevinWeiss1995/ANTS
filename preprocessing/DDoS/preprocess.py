import pandas as pd
import os
from sklearn import preprocessing
import time

# Start timer for processing time
seconds = time.time()

print("This process may take 5 to 10 minutes, depending on the performance of your computer.\n\n\n")

# CSV files names (update these paths if needed to match your repo structure)
csv_files = [
    "Monday-WorkingHours.pcap_ISCX",
    "Tuesday-WorkingHours.pcap_ISCX",
    "Wednesday-workingHours.pcap_ISCX",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
    "Friday-WorkingHours-Morning.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX",
]

# Headers of columns (same as before)
main_labels = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "faulty-Fwd Header Length", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", 
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label", "External IP"
]

# Join column names into a single string
main_labels_str = ",".join(main_labels) + "\n"

flag = True
for i in range(len(csv_files)):
    output_file = os.path.join("ANTS_repo", f"{i}.csv")
    
    with open(output_file, "w") as ths:
        ths.write(main_labels_str)
        
        # Read the CSV file in the CSV folder
        with open(f"./CSVs/{csv_files[i]}.csv", "r") as file:
            while True:
                try:
                    line = file.readline()
                    if line[0] in "0123456789":  # Skip headers and incomplete data
                        # Replace problematic characters
                        line = line.replace(" – ", " - ").replace("inf", "0").replace("Infinity", "0").replace("NaN", "0")
                        ths.write(line)
                except:
                    break
    
    # Load the cleaned CSV into a DataFrame
    df = pd.read_csv(output_file, low_memory=False)
    df = df.fillna(0)

    # Handling non-numeric columns like "Flow Bytes/s" and "Flow Packets/s"
    string_features = ["Flow Bytes/s", "Flow Packets/s"]
    for feature in string_features:
        df[feature] = df[feature].replace("Infinity", -1).replace("NaN", 0)
        df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)

    # Detect non-numeric (string/categorical) columns and encode them
    string_features = [col for col in df.columns if df[col].dtype == "object"]
    if "Label" in string_features:
        string_features.remove("Label")
    
    labelencoder_X = preprocessing.LabelEncoder()
    for feature in string_features:
        df[feature] = labelencoder_X.fit_transform(df[feature].astype(str))

    # Drop the unnecessary column
    df = df.drop(main_labels[61], axis=1)  # Drop the "Fwd Header Length" column if needed

    # Append to a single file after processing each CSV
    if flag:
        df.to_csv('all_data.csv', index=False)
        flag = False
    else:
        df.to_csv('all_data.csv', index=False, header=False, mode="a")
    
    # Remove temporary CSV
    os.remove(output_file)
    
    print(f"The pre-processing phase of the {csv_files[i]} file is completed.\n")

# Display completion message and total operation time
print("mission accomplished!")
print("Total operation time: =", time.time() - seconds, "seconds")
