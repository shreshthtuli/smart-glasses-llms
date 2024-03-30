from src.template import LightningTemplate
from src.utils import get_cuda_device
import torch 
from torch import nn


class FCNNet(LightningTemplate):
    def __init__(self, params=None):
        super().__init__()
        self.input_feat_size = params['input_feat_size']
        self.num_layers = params['num_layers']
        self.hidden_feat_size = params['hidden_feat_size']
        self.dropout = params['dropout']
        layers1 = [nn.Linear(self.input_feat_size, self.hidden_feat_size),
                  nn.ReLU()]
        for _ in range(self.num_layers-1):
            layers1 += [nn.Dropout(self.dropout), 
                       nn.Linear(self.hidden_feat_size, self.hidden_feat_size),
                       nn.LeakyReLU()]
        layers2 = [nn.Linear(self.input_feat_size, self.hidden_feat_size),
                  nn.ReLU()]
        for _ in range(self.num_layers-1):
            layers2 += [nn.Dropout(self.dropout), 
                       nn.Linear(self.hidden_feat_size, self.hidden_feat_size),
                       nn.LeakyReLU()]
        self.net1 = nn.Sequential(*layers1)
        self.net2 = nn.Sequential(*layers2)
        self.readout1 = nn.Sequential(nn.Linear(self.hidden_feat_size, 1), nn.Sigmoid())
        self.readout2 = nn.Sequential(nn.Linear(self.hidden_feat_size, 1), nn.Sigmoid())
        self.save_hyperparameters()

    def forward(self, embedding):
        x = self.net1(embedding)
        pred1 = self.readout1(x)
        x = self.net2(embedding)
        pred2 = self.readout2(x)
        return torch.concat([pred1, pred2], dim=1)   

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)