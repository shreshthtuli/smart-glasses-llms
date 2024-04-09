from ..llmnet import *
from .templateselector import TemplateSelector
from rich.progress import track
import random
from time import time as tt

class RandomSelector(TemplateSelector):
    def __init__(self, DATA_PATH):
        super().__init__(DATA_PATH)
        self.select()
    
    def random_selector(self):
        selection = random.choice(self.all_llm_names)
        return selection

    def select(self):
        print('Running method', self.__class__.__name__)
        max_time = max([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        min_time = min([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        for dset in [self.train_dset, self.test_dset]:
            selections = []; times = []; scores = []; qos = []; stimes = []
            for _, row in track(dset.iterrows(), total=len(dset)):
                start = tt()
                selection = self.random_selector()
                selection_time = tt() - start
                time = row[selection+'_time'] + selection_time
                score = row[selection+'_score']
                q = row['complexity'] * (score-1)/9 + \
                    row['time_criticality'] * (time-min_time)/(max_time-min_time)
                selections.append(selection); times.append(time); scores.append(score)
                qos.append(q); stimes.append(selection_time)
            dset.insert(len(dset.columns), 'selection', selections)
            dset.insert(len(dset.columns), 'selection_time', stimes)
            dset.insert(len(dset.columns), 'time', times)
            dset.insert(len(dset.columns), 'score', scores)
            dset.insert(len(dset.columns), 'qos', qos)
        

