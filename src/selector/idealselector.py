from ..llmnet import *
from .templateselector import TemplateSelector
from glob import glob
import pandas as pd
import numpy as np

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
        for dset in [self.train_dset, self.test_dset]:
            selections = []; times = []; scores = []
            for index, row in dset.iterrows():
                selection = self.ideal_qos_selector(row)
                time = row[selection+'_time']
                score = row[selection+'_score']
                selections.append(selection); times.append(time); scores.append(score)
            dset.insert(len(dset.columns), 'selection', selections)
            dset.insert(len(dset.columns), 'time', times)
            dset.insert(len(dset.columns), 'score', scores)
        

