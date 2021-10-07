# Multitask Deep Learning with Dynamic Task Balancing for Quantum Mechanical Properties Prediction

## Dataset 
download dataset from https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/qm9.zip
and unzip the file to /data/qm9/raw

## Requirements  
torch==1.7.1
torch_sparse==0.6.10
numpy==1.20.1
tqdm==4.51.0
scipy==1.6.2
pandas==1.2.4
torch_geometric==1.7.1
networkx==2.5.1
rdkit==2009.Q1-1
scikit_learn==1.0

## Usage example:  
cd MDGNN
python train.py
