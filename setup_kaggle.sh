#!/bin/bash
# Navigate to the folder (assuming we are in /kaggle/working)
cd LLM-Project/FinDKG

# 1. Create a fresh Python 3.10 environment
conda create -n findkg_env python=3.10 -y

# 2. Activate it 
source /opt/conda/bin/activate findkg_env

# 3. Install PyTorch & DGL
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
pip install pandas numpy networkx scikit-learn tqdm

# 4. Start the training!
python train_DKG_run.py
