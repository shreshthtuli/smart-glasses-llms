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
            title = 'test' if i else 'train'
            os.makedirs(f'{self.plot_path}/{title}/', exist_ok=True)
            for result in ['score', 'time', 'qos']:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_ylabel(result.title())
                sns.violinplot(data=pd.concat(dset.values(), ignore_index=True),
                               x='method', y=result, ax=ax, native_scale=True)
                fig.savefig(f'{self.plot_path}/{title}/{result}.pdf')   
                fig, ax = plt.subplots(figsize=(10, 4))             
                ax.set_ylabel(result.title())
                sns.violinplot(data=pd.concat(dset.values(), ignore_index=True),
                               x='method', y=result, hue='selection', ax=ax, native_scale=True)
                fig.savefig(f'{self.plot_path}/{title}/{result}_selection.pdf')
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.set_ylabel(result.title())
                sns.violinplot(data=pd.concat(dset.values(), ignore_index=True),
                               x='method', y=result, hue='bench', ax=ax, native_scale=True)
                fig.savefig(f'{self.plot_path}/{title}/{result}_bench.pdf')
        