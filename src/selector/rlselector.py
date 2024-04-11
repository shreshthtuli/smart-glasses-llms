from ..rl import *
from ..dataset import CTCDataset
from .templateselector import TemplateSelector
from rich.progress import track
from random import gauss
import pandas as pd
import numpy as np
from pprint import pprint
from time import time as tt
import pickle

import torch
from torchrl.envs import StepCounter, TransformedEnv

torch.manual_seed(42)

class RLSelector(TemplateSelector):
    def __init__(self, MODEL_NAME, MODEL_PATH, DATA_PATH):
        super().__init__(DATA_PATH)
        self.environment = SmartGlassesEnvironment(DATA_PATH)
        self.environment= TransformedEnv(self.environment, StepCounter())
        self.model = eval(MODEL_NAME)(self.environment)
        with open(f'{MODEL_PATH}/{MODEL_NAME}/model.rl', 'rb') as handle:
            self.model.policy = pickle.load(handle)
        self.model_name = MODEL_NAME
        self.select()

    def rl_selector(self, embedding):
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            action = np.argmax(self.model.policy(embedding)[0])
        selection = self.environment.all_llm_names[action]
        return selection

    def select(self):
        print('Running method', self.__class__.__name__, 'with', self.model_name)
        max_time = max([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        min_time = min([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        for dset in [self.train_dset, self.test_dset]:
            selections = []; times = []; scores = []; qos = []; stimes = []
            torch_dset = CTCDataset(dset, model='jinaai/jina-embeddings-v2-base-en')
            for index, row in track(dset.iterrows(), total=len(dset)):
                start = tt()
                embedding, _ = torch_dset.__getitem__(index)
                selection = self.rl_selector(embedding.numpy())
                selection_time = tt() - start
                time = row[selection+'_time'] + selection_time
                score = row[selection+'_score']
                q = row['complexity'] * (score-1)/9 + \
                    row['time_criticality'] * (time-min_time)/(max_time-min_time)
                selections.append(selection); times.append(time); scores.append(score); qos.append(q)
                stimes.append(selection_time)
            dset.insert(len(dset.columns), 'selection', selections)
            dset.insert(len(dset.columns), 'selection_time', stimes)
            dset.insert(len(dset.columns), 'time', times)
            dset.insert(len(dset.columns), 'score', scores)
            dset.insert(len(dset.columns), 'qos', qos)
     
        