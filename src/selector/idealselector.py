from ..llmnet import *
from .templateselector import TemplateSelector
from rich.progress import track
import pandas as pd
import numpy as np
from time import time as tt

class IdealSelector(TemplateSelector):
    def __init__(self, DATA_PATH):
        super().__init__(DATA_PATH)
        self.select()
    
    def ideal_qos_selector(self, row):
        # get percentile times and scores
        max_time, min_time = {}, {}
        for llm_name in self.all_llm_names:
            max_time[llm_name] = self.train_dset[llm_name+'_time'].max()
            min_time[llm_name] = self.train_dset[llm_name+'_time'].min()
        complexity, criticality = row.complexity, row.time_criticality
        qos = [complexity * row[llm_name+'_score'] + 
               criticality * 10 * (1 - ((row[llm_name+'_time'] - min_time[llm_name]) / (max_time[llm_name] - min_time[llm_name]))) 
                for llm_name in self.all_llm_names]
        selection = self.all_llm_names[np.argmax(qos)]
        return selection

    def select(self):
        print('Running method', self.__class__.__name__)
        max_time = max([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        min_time = min([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        for dset in [self.train_dset, self.test_dset]:
            selections = []; times = []; scores = []; qos = []
            for index, row in track(dset.iterrows(), total=len(dset)):
                start = tt()
                selection = self.ideal_qos_selector(row)
                selection_time = tt() - start
                time = row[selection+'_time'] + selection_time
                score = row[selection+'_score']
                q = row['complexity'] * (score-1)/9 + \
                    row['time_criticality'] * (time-min_time)/(max_time-min_time)
                selections.append(selection); times.append(time); scores.append(score)
                qos.append(q)
            dset.insert(len(dset.columns), 'selection', selections)
            dset.insert(len(dset.columns), 'time', times)
            dset.insert(len(dset.columns), 'score', scores)
            dset.insert(len(dset.columns), 'qos', qos)

