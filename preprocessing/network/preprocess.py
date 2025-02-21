import os
import subprocess
import warnings
from math import ceil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pickle

"""
This script takes the all_data.csv file and from data/network and preprocesses it by
dropping all non numeric values, removing columns of all zeros, and renaming label classes
as either Normal or Attack for binary classification. It randomly samples the data to 
address class imbalances. 

"""


# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


def get_git_repo_root():
    """Dynamically determine the Git repository root."""
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        print("Error: Not a git repository.")
        return None
    

base_repo = get_git_repo_root()

data_path = os.path.join(base_repo, "data", "network", "all_data.csv")

pd.options.mode.use_inf_as_na = True

data = pd.read_csv(data_path)

print("Original DataFrame:")
print(data.head())

# Remove identifying and metadata features that could cause data leakage
columns_to_remove = [
    "Flow ID", 
    "Source IP", 
    "Source Port",
    "Destination IP", 
    "Destination Port",
    "Protocol", 
    "Timestamp"
]

data = data.drop(columns=columns_to_remove, errors='ignore')

# Drop columns with all zeros
zero_cols = data.columns[(data == 0).all()]
data_dropped = data.drop(columns=zero_cols)

print("DataFrame after dropping columns with all zeros:")
print(data_dropped.head())

# Print the various labels in data
print(data_dropped.loc[:,"Label"].unique())


trf_type = data_dropped.loc[:, "Label"].map(lambda lbl: "Normal" if lbl == "BENIGN" else "Attack")

print("Labels after binary adjustments: ", trf_type.unique())

trf_type.name = "traffic type"
data_dropped.loc[:, trf_type.name] = trf_type
data_dropped.loc[:, "traffic type"].value_counts()


rus = RandomUnderSampler(random_state=10, sampling_strategy=0.85)
data_dropped.drop(["traffic type"], axis=1, inplace=True)
data_res, trf_type_res = rus.fit_resample(data_dropped, trf_type)
data_sampled = data_res.join(trf_type_res, how="inner")

print("DataFrame after downsampling:")
print(data_sampled.head())


# Prepare data for feature importance 
lbls = data_sampled.loc[:,"Label"]
data_w_o_cat_attrs = data_sampled.iloc[:, :-2]
data_w_o_cat_attrs.reset_index(drop=True, inplace=True)

columns_to_remove = ["Flow ID", "Source IP", "Destination IP", "Protocol", "Timestamp", "Label"]
data_w_o_cat_attrs.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# Remove some of the columns we don't want



# Print out datat with out category attributes
data_w_o_cat_attrs.info()

# Find imporant features
rfc = RandomForestClassifier(random_state=10, n_jobs=-1) 
rfc.fit(data_w_o_cat_attrs, lbls)

score = np.round(rfc.feature_importances_,5)
importances = pd.DataFrame({'features': data_w_o_cat_attrs.columns, 'importance level': score})
importances = importances.sort_values('importance level', ascending=False).set_index('features')

# plot
'''
sns.barplot(x=importances.index, y="importance level", data=importances, color="b")
plt.xticks(rotation="vertical")
plt.gcf().set_size_inches(14,5)
plt.savefig("importances.png", dpi=200, format='png', bbox_inches = "tight", pad_inches=0.2)
plt.show()
'''

# Leave only the most important features 
threshold = 0.001 # importance threshold

bl_thresh = importances.loc[importances["importance level"] < threshold]
print("there are {} features to delete, as they are below the chosen threshold".format(bl_thresh.shape[0]))
print("these features are the following:")
feats_to_del = [feat for feat in bl_thresh.index]
print("\n".join(feats_to_del))

## removing these not important features 
data_sampled.drop(columns=feats_to_del, inplace=True) # dropping columns


# These are highly correlated features
features_to_remove = [
    "Subflow Bwd Packets",
    "Idle Mean",
    "Flow Packets/s",
    "Flow Duration",
    "Total Backward Packets",
    "min_seg_size_forward",
    "Fwd Packet Length Std",
    "Fwd IAT Std",
    "Flow IAT Std",
    "Flow IAT Max",
    "Subflow Fwd Packets",
    "Fwd IAT Max",
    "Idle Min",
    "Total Fwd Packets",
    "Fwd Header Length",
    "Max Packet Length",
    "Total Length of Bwd Packets",
    "Bwd Packet Length Std",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Total Length of Fwd Packets",
    "Bwd Packet Length Mean",
    "Packet Length Mean",
    "Avg Bwd Segment Size",
    "Average Packet Size",
    "External IP"
]

data_sampled.drop(columns=features_to_remove, inplace=True, errors='ignore')

print("Data columns after removing high corr feats: ", data_sampled.columns)


# Scale data
qt = QuantileTransformer(random_state=10)

att_type = data_sampled.loc[:, "Label"]
bin_trff_type = data_sampled.loc[:, "traffic type"]
data_sampled.drop(["Label", "traffic type"], axis=1, inplace=True)  # Drop categorical columns
all_data_scled = qt.fit_transform(data_sampled)

# Save the QuantileTransformer
transformer_dir = os.path.join(base_repo, 'results', 'transformers')
os.makedirs(transformer_dir, exist_ok=True)
transformer_path = os.path.join(transformer_dir, 'quantile_transformer.pkl')

with open(transformer_path, 'wb') as f:
    pickle.dump(qt, f)

print(f"Saved QuantileTransformer to: {transformer_path}")

# Create a DataFrame from the scaled data
scaled_data_df = pd.DataFrame(all_data_scled, columns=data_sampled.columns)

# Add the categorical columns back
scaled_data_df["Label"] = att_type.values
scaled_data_df["traffic type"] = bin_trff_type.values

# Check unique values in the traffic type
print("Unique values in 'traffic type' before encoding:", scaled_data_df["traffic type"].unique())

print("Scaled DataFrame with traffic type added back:")
print(scaled_data_df.head())

# Encode the traffic type
scaled_data_df["traffic type"] = scaled_data_df["traffic type"].map({"Normal": 0, "Attack": 1})

# Print the various labels in data
print(scaled_data_df.loc[:,"traffic type"].unique())

# Check for NaN values after mapping
if scaled_data_df["traffic type"].isnull().any():
    print("Warning: There are NaN values in the encoded traffic type.")

print("Final DataFrame with encoded traffic type:")
print(scaled_data_df.head())

scaled_data_path = os.path.join(base_repo, 'data', 'network', 'scaled_data.csv')
scaled_data_df.to_csv(scaled_data_path, index=False)

print("Scaled and processed data saved successfully.")

# Drop the "Label" column before train_test_split
scaled_data_df.drop(["Label"], axis=1, inplace=True)

# Split the data into training and testing sets
train_data, test_data, train_lbl, test_lbl = train_test_split(scaled_data_df.drop("traffic type", axis=1), 
                                                              scaled_data_df["traffic type"], 
                                                              random_state=10, 
                                                              train_size=0.7)

# Print the heads of the resulting DataFrames
print("Train Data:")
print(train_data.head())

print("Test Data:")
print(test_data.head())

print("Train Labels:")
print(train_lbl.head())

print("Test Labels:")
print(test_lbl.head())

# Save training and test data to CSV files
train_data_path = os.path.join(base_repo, 'data', 'network', 'train_data.csv')
test_data_path = os.path.join(base_repo, 'data', 'network', 'test_data.csv')
train_labels_path = os.path.join(base_repo, 'data', 'network', 'train_labels.csv')
test_labels_path = os.path.join(base_repo, 'data', 'network', 'test_labels.csv')

# Save the files
train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)
train_lbl.to_csv(train_labels_path, index=False, header=True)
test_lbl.to_csv(test_labels_path, index=False, header=True)

print("Training and test data files saved successfully!")

# -----------------------------------------------------------------
