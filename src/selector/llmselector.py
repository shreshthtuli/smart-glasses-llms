from ..llmnet import *
from ..dataset import CTCDataset
from .templateselector import TemplateSelector
from rich.progress import track
import pandas as pd
import numpy as np

class LLMSelector(TemplateSelector):
    def __init__(self, MODEL_NAME, MODEL_PATH, DATA_PATH, percentile=0.5):
        super().__init__(DATA_PATH)
        self.model = eval(MODEL_NAME).load_from_checkpoint(checkpoint_path=f'{MODEL_PATH}/{MODEL_NAME}/checkpoint.ckpt')
        self.percentile = percentile
        self.select()

    def qos_selector(self, complexity, criticality):
        # get percentile times and scores
        p_time, p_score = {}, {}
        for llm_name in self.all_llm_names:
            p_time[llm_name] = self.train_dset[llm_name+'_time'].quantile(self.percentile)
            p_score[llm_name] = self.train_dset[llm_name+'_score'].quantile(self.percentile)
        max_time = self.train_dset[llm_name+'_time'].max()
        min_time = self.train_dset[llm_name+'_time'].min()
        qos = [complexity * p_score[llm_name] + 
               criticality * 10 * (1 - ((p_time[llm_name] - min_time) / (max_time - min_time))) 
                for llm_name in self.all_llm_names]
        selection = self.all_llm_names[np.argmax(qos)]
        return selection

    def select(self):
        print('Running method', self.__class__.__name__)
        for dset in [self.train_dset, self.test_dset]:
            selections = []; times = []; scores = []
            torch_dset = CTCDataset(dset, model='jinaai/jina-embeddings-v2-base-en')
            for index, row in track(dset.iterrows(), total=len(dset)):
                embedding, _ = torch_dset.__getitem__(index)
                output = self.model.predict(embedding.unsqueeze(0))
                complexity, criticality = output.detach().tolist()[0]
                selection = self.qos_selector(complexity, criticality)
                time = row[selection+'_time']
                score = row[selection+'_score']
                selections.append(selection); times.append(time); scores.append(score)
            dset.insert(len(dset.columns), 'selection', selections)
            dset.insert(len(dset.columns), 'time', times)
            dset.insert(len(dset.columns), 'score', scores)
        
        

