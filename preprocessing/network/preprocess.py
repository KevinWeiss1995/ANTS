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

print(f"Original DataFrame shape: {data.shape}")

all_zeroes_cols = data.columns[(data == 0).all()]
data_dropped = data.drop(columns=all_zeroes_cols)

print(f"DataFrame shape after dropping columns with all zeros: {data_dropped.shape}")

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

print(f"Shape after downsampling: {data_sampled.shape}")


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

# Create a DataFrame from the scaled data
scaled_data_df = pd.DataFrame(all_data_scled, columns=data_sampled.columns)

# Check unique values in the traffic type
print("Unique values in 'traffic type':", bin_trff_type.unique())

# Encode the traffic type
encoded_traffic_type = bin_trff_type.map({"BENIGN": 0, "Attack": 1})

# Check for NaN values after mapping
if encoded_traffic_type.isnull().any():
    print("Warning: There are NaN values in the encoded traffic type.")

# Add the encoded traffic type back to the scaled DataFrame
scaled_data_df["traffic type"] = encoded_traffic_type

print(scaled_data_df.head())  # Optional: Print the first few rows to verify

scaled_data_path = os.path.join(base_repo, 'data', 'network', 'scaled_data.csv')
pd.DataFrame(scaled_data_df).to_csv(scaled_data_path)

print("Scaled and processed data saved successfully.")
'''
### Splitting dataset into training and test sets
train_data, test_data, train_lbl, test_lbl = train_test_split(all_data_scled, att_type, random_state=10, train_size=0.7)

## Additional held-out validation set for evaluating neural networks predicting on the upsampled training set 
## The validation set needs to be split before upsampling
neural_train_data, neural_validation, neural_train_lbl, neural_validation_lbl = train_test_split(
    train_data, train_lbl, random_state=10, train_size=0.8
)  ## Will be shuffled in the same order as train_data above

train_bin_trff_lbl = train_lbl.map(lambda lbl: "Normal" if lbl == "BENIGN" else "Attack")
neural_train_bin_trff_lbl = neural_train_lbl.map(lambda lbl: "Normal" if lbl == "BENIGN" else "Attack")  ## train_lbl for upsampled neural nets
test_bin_trff_lbl = test_lbl.map(lambda lbl: "Normal" if lbl == "BENIGN" else "Attack")
neural_validation_bin_trff_lbl = neural_validation_lbl.map(lambda lbl: "Normal" if lbl == "BENIGN" else "Attack")

# Check the current distribution of each traffic type in training set
a = train_lbl.value_counts()
all_samples = a.sum()
print(a)
print("Total: {}".format(all_samples))


min_thresh = 0.005  # it is a percent of the whole traffic after underSampling

glob_cls_distr = None
def over_sample_new(y):
    global glob_cls_distr
    cls_distr = {}
    for trf_cls in np.unique(y):
        curr_size = a.loc[trf_cls]  # global a == train_lbl.value_counts()
        if (curr_size / all_samples) < min_thresh:
            cls_distr[trf_cls] = ceil(min_thresh * all_samples)
        else:
            cls_distr[trf_cls] = curr_size
    print("class distribution after over sampling:")
    glob_cls_distr = cls_distr
    print(glob_cls_distr)
    return cls_distr

def over_sample_bin(dct):
    sm = 0
    for key, val in dct.items():
        if key != "BENIGN":
            sm += val
        else: benign = val
    return {"Normal": benign, "Attack": sm}

#dct = {'FTP-Patator': 7935, 'SSH-Patator': 6057, 'DoS slowloris': 6057, 'DoS Slowhttptest': 6057, 'DoS Hulk': 230124, 'DoS GoldenEye': 10293, 'Heartbleed': 6057, 'Brute Force': 6057, 'XSS': 6057, 'Sql Injection': 6057, 'Infiltration': 6057, 'DDoS': 128025, 'PortScan': 158804, 'Bot': 6057, 'BENIGN': 654771}
smote = SMOTE(random_state=10, k_neighbors=3, sampling_strategy=over_sample_new)  # todo can resample w/ k_neigh only for heartbleed
#print(glob_cls_distr)

up_train_data, up_train_lbl = smote.fit_resample(train_data, train_lbl)
 
ratio = over_sample_bin(glob_cls_distr)
#print(ratio)
smote_bin = SMOTE(random_state=10, k_neighbors=3, sampling_strategy=ratio)
up_train_bin_data, up_train_bin_trff_lbl = smote_bin.fit_resample(train_data, train_bin_trff_lbl)

# Label encoding

test_rshped = test_lbl.values.reshape(-1,1)
train_rshped = train_lbl.values.reshape(-1,1)
up_train_rshped = up_train_lbl.values.reshape(-1,1)

ohenc = OneHotEncoder()
lenc = LabelEncoder()

test_lbl_enc = ohenc.fit_transform(test_rshped).toarray()  # one-hot encoded test set lbls
train_lbl_enc = ohenc.fit_transform(train_rshped).toarray()  # one-hot encoded train set labels
up_train_lbl_enc = ohenc.fit_transform(up_train_rshped).toarray()  # one-hot encoded upsampled train set lbls


test_bin_trff_lbll_enc = lenc.fit_transform(test_bin_trff_lbl)  # label encoded test set binary lbls
train_bin_trff_lbll_enc = lenc.fit_transform(train_bin_trff_lbl) # label encoded train set binary lbls
up_train_bin_trff_lbl_enc = lenc.fit_transform(up_train_bin_trff_lbl)  # label encoded upsampled train set binary lbls


# Save label-encoded binary labels as CSV files

train_bin_trff_lbll_enc_path = os.path.join(base_repo, 'data', 'network', 'train_bin_trff_lbll_enc.csv')
test_bin_labels_path = os.path.join(base_repo, 'data', 'network', 'test_bin_labels.csv')
up_train_bin_labels_path = os.path.join(base_repo, 'data', 'network', 'up_train_bin_labels.csv')


pd.DataFrame(train_bin_trff_lbll_enc).to_csv(train_bin_trff_lbll_enc_path, index=False, header=False)
pd.DataFrame(test_bin_trff_lbll_enc).to_csv(test_bin_labels_path, index=False, header=False)
pd.DataFrame(up_train_bin_trff_lbl_enc).to_csv(up_train_bin_labels_path, index=False, header=False)

print("Binary label files saved successfully!")
'''