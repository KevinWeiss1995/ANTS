import random
import os
import pandas as pd
import time

seconds = time.time()

# Function to create a folder if it does not exist
def folder(f_name):
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError as e:
        print(f"Error creating folder {f_name}: {e}")

# Print an informative message about the processing time
print("This process may take 3 to 8 minutes, depending on the performance of your computer.\n\n\n")

# Headers of columns
main_labels = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", 
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
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label", "External IP"
]
main_labels = ",".join(main_labels)

# Attack types and number of occurrences
attacks = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris", "FTP-Patator", 
    "Heartbleed", "Infiltration", "PortScan", "SSH-Patator", "Web Attack – Brute Force", "Web Attack – Sql Injection", 
    "Web Attack – XSS"
]

# Create output folder for attack data if it doesn't exist
folder("./attacks/")

# Benign flows
benign = 2359289

# Dictionary with attack types and the corresponding number of occurrences
dict_attack = {
    "Bot": 1966, "DDoS": 41835, "DoS GoldenEye": 10293, "DoS Hulk": 231073, "DoS Slowhttptest": 5499, 
    "DoS slowloris": 5796, "FTP-Patator": 7938, "Heartbleed": 11, "Infiltration": 36, "PortScan": 158930, 
    "SSH-Patator": 5897, "Web Attack - Brute Force": 1507, "Web Attack - XSS": 652, "Web Attack - Sql Injection": 21
}

# Process each attack type
for attack_type, attack_count in dict_attack.items():
    a, b = 0, 0  # Counters for attacks and benign samples
    # Open a file for the attack type
    with open(f"./attacks/{attack_type}.csv", "w") as ths:
        ths.write(f"{main_labels}\n")
        benign_num = int(benign / (attack_count * (7 / 3)))  # Number of benign samples per attack type
        # Read the full data file and process
        with open("all_data.csv", "r") as file:
            while True:
                try:
                    line = file.readline().strip()
                    k = line.split(",")
                    # If it's a benign flow, randomly choose whether to include it
                    if k[83] == "BENIGN":
                        if random.randint(1, benign_num) == 1:
                            ths.write(f"{line}\n")
                            b += 1
                    elif k[83] == attack_type:  # If it matches the attack type, write it
                        ths.write(f"{line}\n")
                        a += 1
                except Exception as e:
                    print(f"Error reading line: {e}")
                    break
    print(f"{attack_type} file completed\nAttack: {a}\nBenign: {b}\n\n")

# Combine all web attack files into a single file
webs = ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"]
for web_attack in webs:
    df = pd.read_csv(f"./attacks/{web_attack}.csv")
    mode = 'a' if os.path.exists('./attacks/Web Attack.csv') else 'w'
    df.to_csv('./attacks/Web Attack.csv', index=False, header=not os.path.exists('./attacks/Web Attack.csv'), mode=mode)
    os.remove(f"./attacks/{web_attack}.csv")

print("Mission accomplished!")
print(f"Operation time: {time.time() - seconds:.2f} seconds")
