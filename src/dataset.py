import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoModel

# Complexity / Time-Criticality Dataset
class CTCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, model: str = 'jinaai/jina-embeddings-v2-small-en'):
        self.df = df 
        self.transformer = AutoModel.from_pretrained(model, trust_remote_code=True)

    def __len__(self):
        return len(self.df)
    
    def get_embedding_size(self):
        return self.transformer.encode('hi').size

    def __getitem__(self, idx, df=None):
        # get embedding
        instruction = self.df.iloc[idx].turns[0]
        embedding = torch.tensor(self.transformer.encode(instruction))
        # get scores
        complexity = self.df.iloc[idx].complexity
        time_criticality = self.df.iloc[idx].time_criticality
        return embedding, torch.tensor([complexity, time_criticality], dtype=torch.float32)
