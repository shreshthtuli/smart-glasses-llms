from src.template import LightningTemplate
from src.utils import get_cuda_device
import torch 
from torch import nn
from torch.nn import Transformer
torch.manual_seed(42)

class EENet(LightningTemplate):
    def __init__(self, params=None):
        super().__init__()
        self.input_feat_size = params['input_feat_size']
        self.n_heads = params['n_heads']
        self.transformer_layers = params['transformer_layers']
        self.transformer_hidden_size = params['transformer_hidden_size']
        self.linear_hidden_size = params['linear_hidden_size']
        self.num_exits = params['num_exits']
        self.dropout = params['dropout']
        self.complexity_encoder = nn.Sequential( 
            nn.Linear(self.input_feat_size, self.transformer_hidden_size), nn.Sigmoid()
        )
        self.attn = Transformer(d_model=self.transformer_hidden_size, 
                                nhead=self.n_heads, 
                                num_encoder_layers=self.transformer_layers,
                                num_decoder_layers=self.transformer_layers,
                                dropout=self.dropout,
                                activation='gelu')
        self.complexity_decoder = nn.Sequential( 
            nn.Linear(self.transformer_hidden_size, 1), nn.Sigmoid()
        )
        self.criticality_encoder = nn.Sequential(
                nn.Linear(self.input_feat_size, self.linear_hidden_size), nn.LeakyReLU(),
            )
        self.criticality, self.exit, self.classifier = [], [], []
        for _ in range(self.num_exits):
            self.criticality.append(nn.Sequential(
                nn.Linear(self.linear_hidden_size, self.linear_hidden_size), nn.LeakyReLU(),
                nn.Linear(self.linear_hidden_size, self.linear_hidden_size), nn.LeakyReLU(),
            ))
            self.exit.append(nn.Sequential(
                nn.Linear(self.linear_hidden_size, 1), nn.Sigmoid()
            ))
            self.classifier.append(nn.Sequential(
                nn.Linear(self.linear_hidden_size + 1, self.linear_hidden_size), nn.LeakyReLU(),
                nn.Linear(self.linear_hidden_size, 1), nn.Sigmoid()
            ))
        self.save_hyperparameters()

    def complexity(self, embedding):
        x = self.complexity_encoder(embedding)
        x = x.unsqueeze(0)
        x = self.attn(x, x)
        x = x.squeeze(0)
        x = self.complexity_decoder(x)
        return x

    def forward(self, embedding):
        complexity = self.complexity(embedding)
        encoded = self.criticality_encoder(embedding)
        criticality_results = []
        for i in range(len(self.criticality)):
            pred = self.exit[i](self.criticality[i](encoded))
            cl = self.classifier[i](torch.cat([complexity, self.criticality[i](encoded)], 
                                                 dim=1))
            criticality_results.append((pred, cl))
        return complexity, criticality_results

    def step(self, batch):
        inputs, labels = batch
        complexity, criticality_results = self.forward(inputs)
        c_loss = nn.functional.mse_loss(complexity.view(-1), labels[:, 0])
        t_loss = 0; e_loss = 0
        # print("pred complexity", complexity.view(-1).detach().numpy())
        # print("complexity", labels[:, 0].numpy())
        for i, (pred, cl) in enumerate(criticality_results):
            t_loss += nn.functional.mse_loss(pred.view(-1), labels[:, 1])
            gt_cl = torch.round(pred.detach().view(-1) * 5) == (labels[:, 1]*5) # same or neighboring class (1 to 10)
            e_loss += nn.functional.binary_cross_entropy(cl.view(-1), gt_cl.type(torch.float))
            # print(f"{i} pred criticality", pred.view(-1).detach().numpy(), cl.view(-1).detach().numpy())
            # print(f"{i} criticality", labels[:, 1].numpy(), gt_cl.numpy())
        loss = c_loss + t_loss + e_loss
        return loss 
    
    def training_step(self, batch):
        loss = self.step(batch)
        inputs, labels = batch
        self.log("train_loss", loss, on_epoch=True, logger=True)
        outputs = self.predict(inputs)
        self.metric_stack(preds=outputs, gts=labels, subset='train')
        return loss

    def validation_step(self, batch):
        loss = self.step(batch)
        inputs, labels = batch
        self.log("val_loss", loss, on_epoch=True, logger=True)
        outputs = self.predict(inputs)
        self.metric_stack(preds=outputs, gts=labels, subset='valid')
        return loss

    def test_step(self, batch):
        inputs, labels = batch
        outputs = self.predict(inputs)
        self.metric_stack(preds=outputs, gts=labels, subset='test')

    def predict(self, inputs):
        outputs = []
        with torch.no_grad():
            for embedding in inputs:
                complexity = self.complexity(embedding)
                encoded = self.criticality_encoder(embedding)
                for i in range(len(self.criticality)):
                    cl = self.classifier[i](torch.cat([complexity, self.criticality[i](encoded)]))
                    if i != len(self.criticality)-1 and cl.item() <= 0.5:
                        continue
                    criticality = self.exit[i](self.criticality[i](encoded))
                    break
                outputs.append(torch.tensor([complexity, criticality]))
        return torch.stack(outputs).detach()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)

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
    
