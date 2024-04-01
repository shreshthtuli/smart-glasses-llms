from ..llmnet import *
from .templateselector import TemplateSelector
from rich.progress import track
import random

class RandomSelector(TemplateSelector):
    def __init__(self, DATA_PATH):
        super().__init__(DATA_PATH)
        self.select()
    
    def random_selector(self):
        selection = random.choice(self.all_llm_names)
        return selection

    def select(self):
        print('Running method', self.__class__.__name__)
        for dset in [self.train_dset, self.test_dset]:
            selections = []; times = []; scores = []
            for _, row in track(dset.iterrows(), total=len(dset)):
                selection = self.random_selector()
                time = row[selection+'_time']
                score = row[selection+'_score']
                selections.append(selection); times.append(time); scores.append(score)
            dset.insert(len(dset.columns), 'selection', selections)
            dset.insert(len(dset.columns), 'time', times)
            dset.insert(len(dset.columns), 'score', scores)
        

