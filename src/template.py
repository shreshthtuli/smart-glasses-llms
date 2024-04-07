import lightning.pytorch as lightpl
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance, pairwise_minkowski_distance
from torchmetrics import MeanSquaredError, PearsonCorrCoef, ConcordanceCorrCoef, ExplainedVariance, KendallRankCorrCoef
from src.utils import get_cuda_device
import torch 
from torch import nn


class LightningTemplate(lightpl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cuda_device = get_cuda_device()
        self.task = 'binary'
        self.pcs = lambda x: pairwise_cosine_similarity(x[0], x[1], reduction='mean')        
        self.ped = lambda x: pairwise_euclidean_distance(x[0], x[1], reduction='mean')
        self.pmd = lambda x: pairwise_minkowski_distance(x[0], x[1], reduction='mean', exponent=2)
        self.mse = MeanSquaredError()
        self.pcc = PearsonCorrCoef(num_outputs=2)
        self.ccc = ConcordanceCorrCoef(num_outputs=2)
        self.krc = KendallRankCorrCoef(num_outputs=2)
        self.ev  = ExplainedVariance()
        self.model = None

    def metric_stack(self, preds, gts, subset):
        self.log(f'{subset}/PairwiseCosineSimilarity', self.pcs([preds, gts]).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/PairwiseEuclideanDistance', self.ped([preds, gts]).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/PairwiseMinkowskiDistance', self.pmd([preds, gts]).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/MeanSquaredError', self.mse(preds, gts), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/PearsonCorrCoef', self.pcc(preds, gts).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/ConcordanceCorrCoef', self.ccc(preds, gts).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/ExplainedVariance', self.ev(preds, gts), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{subset}/KendallRankCorrCoef', self.krc(preds, gts).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = nn.functional.mse_loss(outputs, labels)
        self.log("train_loss", loss, on_epoch=True, logger=True)
        self.metric_stack(preds=outputs, gts=labels, subset='train')
        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = nn.functional.mse_loss(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, logger=True)
        self.metric_stack(preds=outputs, gts=labels, subset='valid')
        return loss

    def test_step(self, batch):
        inputs, labels = batch
        outputs = self.forward(inputs)
        self.metric_stack(preds=outputs, gts=labels, subset='test')

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs.detach()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)