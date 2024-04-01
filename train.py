from torch.utils.data import DataLoader

import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar

from src.utils import parse_args, get_cuda_device
from src.dataset import CTCDataset
from src.llmnet import FCNNet

import pandas as pd
import shutil

if __name__ == '__main__':
    args = parse_args()
    MODEL = args.model
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 4
    DATA_PATH = f'./data/'
    MODEL_SAVE_PATH = f'./models/{MODEL}/'
    METRIC = 'val_loss'

    shutil.rmtree(f'./logs/{MODEL}', ignore_errors=True)

    # load data
    train_dset = pd.read_json(f'{DATA_PATH}/train.jsonl', lines=True)
    test_dset = pd.read_json(f'{DATA_PATH}/test.jsonl', lines=True)
    train_dset = CTCDataset(train_dset, model='jinaai/jina-embeddings-v2-base-en')
    test_dset = CTCDataset(test_dset, model='jinaai/jina-embeddings-v2-base-en')
    train = DataLoader(train_dset, 
                       batch_size=BATCH_SIZE, 
                       num_workers=NUM_WORKERS, 
                       persistent_workers=True)
    test = DataLoader(test_dset, 
                       batch_size=BATCH_SIZE, 
                       num_workers=NUM_WORKERS, 
                       persistent_workers=True)

    # instantiate model
    model = eval(MODEL)(params={'input_feat_size': train_dset.get_embedding_size(), 
                                'num_layers': 3, 
                                'hidden_feat_size': 512, 
                                'dropout': 0.1})

    # lightning trainer
    trainer = pl.Trainer(accelerator=get_cuda_device(),
                            deterministic=True,
                            devices="auto",
                            num_sanity_val_steps=0,
                            default_root_dir=f'./logs/{MODEL}',
                            max_epochs=200,
                            enable_checkpointing=False,
                            callbacks=[EarlyStopping(monitor=METRIC,
                                                    patience=5, verbose=True,
                                                    mode="min"),
                                    RichProgressBar(refresh_rate=3, leave=True),
                                    ]
                            )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=test)
    trainer.save_checkpoint(MODEL_SAVE_PATH+f'checkpoint.ckpt')