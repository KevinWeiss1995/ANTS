#  ANTS 
## Automated Network Threat Detection and Mitigation

## Overview  
**Automated Network Threat Detection and Mitigation** is a comprehensive solution designed to identify, analyze, and respond to network threats. This system leverages advanced machine learning models to detect malicious activities such as intrusion attempts, phishing, malware, and botnet behavior. It is built to process data from various sources like packet captures (PCAP), log files, email content, and endpoint telemetry.

## Models

DDoS: Random Forest Classifier - Random Forest can handle high dimensional data and is resilient to noise and overfitting, making it ideal for detecting DDoS attacks. They are also robust against imbalanced datasets, which in our case is essential. 

Malware: Graph Neural Network - Through graph representation of data we can model the complex relationships and interactions malware often exhibits (file relationships, network traffic, etc). GNNs can aggregate information from neighboring nodes, allowing them to learn both local and global features of the graph. They are also capable of handling irregular data structures that traditional machine learning models struggle with. 

Phising: ? (Python Libraries: Libraries like BeautifulSoup for parsing HTML, requests for fetching URLs, and email for handling email data can be useful for feature extraction)

## Preprocessing
For DDoS, run in this order:

1) preprocess.py
2) filter_attacks.py
3) feature_select_attack.py
4) feature_select_all.py

For malware:

1) preprocess.py
2) gnn_preprocess.py

## (Future) Features 
- **Multi-Model Threat Detection**: Implements a variety of machine learning models for detecting diverse threat vectors, including:
  - Distributed Denial-of-Service (DDoS) attacks.
  - Port scans.
  - Web-based and phishing attacks.
  - Malware and botnet activity.
- **Automated Response**: Deploys intelligent responses to neutralize threats in real-time, such as blocking malicious IPs and quarantining compromised systems.
- **Continuous Monitoring**: Continuously ingests and analyzes network traffic and system telemetry to ensure robust protection.
- **Data Preprocessing**: Includes tools for cleaning, normalizing, and extracting features from datasets for efficient model training and prediction.
- **Scalable Architecture**: Designed to operate in high-throughput environments, supporting large-scale networks and datasets.

## Data Sources  
The system is trained and tested using a variety of real-world and synthetic datasets, including:  
- Packet captures (PCAP)  
- Network flow logs  
- System telemetry and endpoint data  
- Threat intelligence feeds

Data obtained from 

DDos: https://www.unb.ca/cic/datasets/

Malware: https://github.com/imfaisalmalik/CTU13-CSV-Dataset

Phishing: https://archive.ics.uci.edu/dataset/327/phishing+websites

In base directory, create data/malware and data/DDoS and save the respective data there.

## Requirements

```bash
pip3 install -r requirements.txt
```



## ToDo

* Train GNN with GPU once resources arrive 
* Set up WSGI
* Data pipeline
* Deploy again botnet 


