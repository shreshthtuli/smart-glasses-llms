import lightning.pytorch as lightpl
from torchmetrics.regression import ConcordanceCorrCoef, CosineSimilarity, CriticalSuccessIndex, KendallRankCorrCoef
from torchmetrics import ExplainedVariance
from src.utils import get_cuda_device
import torch 
from torch import nn


class LightningTemplate(lightpl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cuda_device = get_cuda_device()
        self.task = 'binary'
        self.ccf = ConcordanceCorrCoef(num_outputs=2)        
        self.cs = CosineSimilarity(reduction='mean')
        self.csi = CriticalSuccessIndex(threshold=0.5)
        self.krc = KendallRankCorrCoef(num_outputs=2)
        self.ev = ExplainedVariance()
        self.model = None

    def metric_stack(self, preds, gts, subset):
        self.log(f'{subset}/ConcordanceCorrCoef', self.ccf(preds, gts).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/CosineSimilarity', self.cs(preds, gts), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/CriticalSuccessIndex', self.csi(preds, gts), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/KendallRankCorrCoef', self.krc(preds, gts).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/ExplainedVariance', self.ev(preds, gts), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels)
        self.log("train_loss", loss, on_epoch=True, logger=True)
        self.metric_stack(preds=outputs, gts=labels, subset='train')
        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, logger=True)
        self.metric_stack(preds=outputs, gts=labels, subset='valid')
        return loss

    def test_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        self.metric_stack(preds=outputs, gts=labels, subset='test')

    def predict(self, batch):
        inputs, _ = batch
        outputs = self.forward(inputs)
        return outputs.detach()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)