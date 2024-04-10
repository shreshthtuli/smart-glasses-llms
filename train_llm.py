from torch.utils.data import DataLoader, SubsetRandomSampler

import lightning as pl
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint

from sklearn.model_selection import train_test_split

from optuna.integration import PyTorchLightningPruningCallback

from src.utils import parse_args, get_cuda_device
from src.dataset import CTCDataset
from src.llmnet import FCNNet, EENet

from rich.progress import track

import pandas as pd
import optuna
import shutil

if __name__ == '__main__':
    args = parse_args()
    MODEL = args.model
    BATCH_SIZE = args.batch_size
    NUM_TRIALS = args.num_trials
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
    train_dset_tmp, val_dset_tmp = train_test_split(train_dset, test_size=0.1)
    train = DataLoader(train_dset_tmp,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    persistent_workers=True)
    val = DataLoader(val_dset_tmp,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    persistent_workers=True)
    test = DataLoader(test_dset, 
                       batch_size=BATCH_SIZE, 
                       num_workers=NUM_WORKERS, 
                       persistent_workers=True)

    def objective(trial: optuna.trial.Trial):
        if args.model == 'FCNNet':
            num_layers = trial.suggest_categorical("num_layers", [2, 3, 4])
            hidden_feat_size = trial.suggest_categorical("hidden_feat_size", [256, 512, 1024])
            dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5])
            params = dict(num_layers=num_layers, hidden_feat_size=hidden_feat_size, dropout=dropout)
        if args.model == 'EENet':
            n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
            transformer_layers = trial.suggest_categorical("transformer_layers", [1, 2, 3, 4])
            transformer_hidden_size = trial.suggest_categorical("transformer_hidden_size", [32, 64, 128, 256])
            linear_hidden_size = trial.suggest_categorical("linear_hidden_size", [32, 64, 128, 256])
            num_exits = trial.suggest_categorical("num_exits", [4])
            dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5])
            params = dict(n_heads=n_heads, transformer_layers=transformer_layers,
                          linear_hidden_size=linear_hidden_size, num_exits=num_exits,
                          transformer_hidden_size=transformer_hidden_size, dropout=dropout)        
        params["input_feat_size"] = train_dset.get_embedding_size()

        # instantiate model
        model = eval(MODEL)(params=params)

        patience = 50 if args.model == 'EENet' else 5
        trainer = pl.Trainer(num_sanity_val_steps=0,
                             deterministic=True,
                             accelerator=get_cuda_device(),
                             default_root_dir=f'./logs/{MODEL}',
                             callbacks=[EarlyStopping(monitor=METRIC,
                                                      patience=patience, 
                                                      verbose=True,
                                                      mode="min"),
                                        ModelCheckpoint(f'./logs/{MODEL}',
                                                        filename='best_model',
                                                        monitor=METRIC, 
                                                        save_top_k=1),
                                        RichProgressBar(refresh_rate=3, leave=False),
                                        PyTorchLightningPruningCallback(trial, monitor="val_loss")]
                                        )
        trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
        results = trainer.validate(ckpt_path='best', dataloaders=val)[0]
        if trial.number == 0 or results[METRIC] <= trial.study.best_value:
            trainer.save_checkpoint(MODEL_SAVE_PATH+f'checkpoint.ckpt')
        return results[METRIC]

    # do the optuna study
    study = optuna.create_study(direction="minimize",
                                storage='sqlite:///db.sqlite3', 
                                study_name=f'{MODEL}',
                                load_if_exists=True)
    if len(study.trials) != NUM_TRIALS:
        study.optimize(objective, n_trials=NUM_TRIALS)

    # training best model on complete dataset
    best_params = study.best_trial.params
    best_params['input_feat_size'] = train_dset.get_embedding_size()
    model = eval(MODEL)(params=best_params)
    train_complete = DataLoader(train_dset, 
                       batch_size=BATCH_SIZE, 
                       num_workers=NUM_WORKERS, 
                       persistent_workers=True)
    trainer = pl.Trainer(num_sanity_val_steps=0,
                         max_epochs=200,
                         accelerator=get_cuda_device(),
                         default_root_dir=f'./final_logs/{MODEL}',
                         enable_checkpointing=False,
                         callbacks=[EarlyStopping(monitor=METRIC,
                                                    patience=5, verbose=True,
                                                    mode="min"),
                                    RichProgressBar(refresh_rate=3, leave=True),
                                    ]
                            )
    trainer.fit(model, train_dataloaders=train_complete, val_dataloaders=test)
    trainer.save_checkpoint(MODEL_SAVE_PATH+f'checkpoint.ckpt')