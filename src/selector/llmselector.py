from ..llmnet import *
from ..dataset import CTCDataset
from .templateselector import TemplateSelector
from rich.progress import track
from random import gauss
import pandas as pd
import numpy as np
from pprint import pprint
from time import time as tt
import torch
torch.manual_seed(42)

class LLMSelector(TemplateSelector):
    def __init__(self, MODEL_NAME, MODEL_PATH, DATA_PATH, exit=None, percentile=0.5):
        super().__init__(DATA_PATH)
        self.model = eval(MODEL_NAME).load_from_checkpoint(checkpoint_path=f'{MODEL_PATH}/{MODEL_NAME}/checkpoint.ckpt')
        self.model.eval()
        self.exit = exit
        self.percentile = percentile
        self.model_name = MODEL_NAME
        self.select()

    def qos_selector(self, complexity, criticality):
        # get percentile times and scores
        p_time, p_score = {}, {}
        for llm_name in self.all_llm_names:
            p_time[llm_name] = gauss(self.train_dset[llm_name+'_time'].quantile(self.percentile), 
                                     0.2*self.train_dset[llm_name+'_time'].std())
            p_score[llm_name] = gauss(self.train_dset[llm_name+'_score'].quantile(self.percentile), 
                                      0.2*self.train_dset[llm_name+'_score'].std())
        max_time = max([self.train_dset[llm_name+'_time'].max() for llm_name in self.all_llm_names])
        min_time = min([self.train_dset[llm_name+'_time'].min() for llm_name in self.all_llm_names])
        qos = [complexity * 10 * ((p_score[llm_name] - 1) / 9) + 
               criticality * 10 * (1 - ((p_time[llm_name] - min_time) / (max_time - min_time))) 
                for llm_name in self.all_llm_names]
        selection = self.all_llm_names[np.argmax(qos)]
        # qosd = dict(zip(self.all_llm_names, qos))
        # res = [(qosd[llm_name], 10 * ((p_score[llm_name] - 1) / 9), 10 * (1 - ((p_time[llm_name] - min_time) / (max_time - min_time))) )
        #        for llm_name in self.all_llm_names]
        # pprint(dict(zip(self.all_llm_names, res)))
        # print(selection)
        return selection

    def select(self):
        print('Running method', self.__class__.__name__, 'with', self.model_name)
        max_time = max([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        min_time = min([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        for dset in [self.train_dset, self.test_dset]:
            selections = []; times = []; scores = []; qos = []; stimes = []
            pred_complexity = []; pred_time_criticality = []
            torch_dset = CTCDataset(dset, model='jinaai/jina-embeddings-v2-base-en')
            for index, row in track(dset.iterrows(), total=len(dset)):
                start = tt()
                embedding, _ = torch_dset.__getitem__(index)
                output = self.model.predict(embedding.unsqueeze(0)) if self.exit is None else self.model.predict1(embedding.unsqueeze(0))
                complexity, criticality = output.detach().tolist()[0]
                selection = self.qos_selector(complexity, criticality)
                selection_time = tt() - start
                time = row[selection+'_time'] + selection_time
                score = row[selection+'_score']
                q = row['complexity'] * (score-1)/9 + \
                    row['time_criticality'] * (time-min_time)/(max_time-min_time)
                selections.append(selection); times.append(time); scores.append(score); qos.append(q)
                pred_complexity.append(complexity); pred_time_criticality.append(criticality)
                stimes.append(selection_time)
            dset.insert(len(dset.columns), 'pred_complexity', pred_complexity)
            dset.insert(len(dset.columns), 'pred_time_criticality', pred_time_criticality)            
            dset.insert(len(dset.columns), 'selection', selections)
            dset.insert(len(dset.columns), 'selection_time', stimes)
            dset.insert(len(dset.columns), 'time', times)
            dset.insert(len(dset.columns), 'score', scores)
            dset.insert(len(dset.columns), 'qos', qos)
     
        