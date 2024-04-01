import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scienceplots
import os

plt.style.use(['science'])

class Plotter():
    def __init__(self, PLOT_PATH='./plots/'):
        self.train_results = {}
        self.test_results = {}
        self.plot_path = PLOT_PATH
        os.makedirs(PLOT_PATH, exist_ok=True)
    
    def add_results(self, method, train_dset, test_dset):
        self.train_results[method] = train_dset.copy(deep=True)
        self.test_results[method] = test_dset.copy(deep=True)
        self.train_results[method]['method'] = method
        self.test_results[method]['method'] = method
    
    def plot_scores_times(self):
        for i, dset in enumerate([self.train_results, self.test_results]):
            for result in ['score', 'time']:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_ylabel(result.title())
                sns.violinplot(data=pd.concat(dset.values(), ignore_index=True),
                               x='method', y=result, ax=ax)
                title = 'test' if i else 'train'
                fig.savefig(f'{self.plot_path}{result}_{title}.pdf')   
                fig, ax = plt.subplots(figsize=(10, 4))             
                ax.set_ylabel(result.title())
                sns.violinplot(data=pd.concat(dset.values(), ignore_index=True),
                               x='method', y=result, hue='selection', ax=ax)
                title = 'test' if i else 'train'
                fig.savefig(f'{self.plot_path}{result}_{title}_selection.pdf')
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.set_ylabel(result.title())
                sns.violinplot(data=pd.concat(dset.values(), ignore_index=True),
                               x='method', y=result, hue='bench', ax=ax)
                title = 'test' if i else 'train'
                fig.savefig(f'{self.plot_path}{result}_{title}_bench.pdf')
        