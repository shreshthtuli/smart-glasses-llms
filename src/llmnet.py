from src.template import LightningTemplate
from torchmetrics import F1Score, Precision, Recall, MatthewsCorrCoef, \
    Accuracy, AUROC, AveragePrecision, ConfusionMatrix, CohenKappa
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
        layers = [nn.Linear(self.input_feat_size, self.hidden_feat_size),
                  nn.ReLU()]
        for _ in range(self.num_layers-1):
            layers += [nn.Dropout(self.dropout), 
                       nn.Linear(self.hidden_feat_size, self.hidden_feat_size),
                       nn.LeakyReLU()]
        self.net = nn.Sequential(*layers)
        self.readout = nn.Sequential(nn.Linear(self.hidden_feat_size, 2), nn.Sigmoid())
        self.save_hyperparameters()

    def forward(self, embedding):
        x = self.net(embedding)
        pred = self.readout(x)
        return pred   

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)