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
        ylabel = {'score': 'Score', 'time': 'Time', 'qos': 'QoS'}
        methods = {'FCN': 'SELA w/o EE', 'EE': 'SELA'}
        for i, dset in enumerate([self.train_results, self.test_results]):
            title = 'test' if i else 'train'
            os.makedirs(f'{self.plot_path}/{title}/', exist_ok=True)
            for result in ['score', 'time', 'qos']:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_ylabel(ylabel[result])
                sns.boxplot(data=pd.concat(dset.values(), ignore_index=True), whis=(0,100),
                               x='method', y=result, ax=ax, native_scale=True)
                plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=0.)
                ax.set_xlabel('Method')
                ax.set_xticklabels(labels=[(methods[i] if i in methods else i) for i in [j._text for j in ax.get_xticklabels()]])
                plt.xticks(rotation=90)
                fig.savefig(f'{self.plot_path}/{title}/{result}.pdf')   
                fig, ax = plt.subplots(figsize=(10, 4))             
                ax.set_ylabel(ylabel[result])
                sns.boxplot(data=pd.concat(dset.values(), ignore_index=True), whis=(0,100),
                               x='method', y=result, hue='selection', ax=ax, native_scale=True)
                plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=0.)
                ax.set_xlabel('Method')
                ax.set_xticklabels(labels=[(methods[i] if i in methods else i) for i in [j._text for j in ax.get_xticklabels()]])
                fig.savefig(f'{self.plot_path}/{title}/{result}_selection.pdf')
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.set_ylabel(ylabel[result])
                sns.boxplot(data=pd.concat(dset.values(), ignore_index=True), whis=(0,100),
                               x='method', y=result, hue='bench', ax=ax, native_scale=True)
                plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=0.)
                ax.set_xlabel('Method')
                ax.set_xticklabels(labels=[(methods[i] if i in methods else i) for i in [j._text for j in ax.get_xticklabels()]])
                fig.savefig(f'{self.plot_path}/{title}/{result}_bench.pdf')
            
    def gen_scores_times_table(self):
        methods = {'FCN': 'SELA w/o EE', 'EE': 'SELA'}
        for i, dset in enumerate([self.train_results, self.test_results]):
            title = 'test' if i else 'train'
            records = []
            for method, df in dset.items():
                record = {'method': methods[method] if method in methods else method}
                for bench in df.bench.drop_duplicates().sort_values()[:2]:
                    df_bench = df[df.bench == bench]
                    record[f'{bench}_st'] = f'{df_bench.selection_time.mean():.3f}$\pm${df_bench.selection_time.std():.3f}'
                    record[f'{bench}_time'] = f'{df_bench.time.mean():.3f}$\pm${df_bench.time.std():.3f}'
                    record[f'{bench}_score'] = f'{df_bench.score.mean():.3f}$\pm${df_bench.score.std():.3f}'
                    record[f'{bench}_qos'] = f'{df_bench.qos.mean():.3f}$\pm${df_bench.qos.std():.3f}'
                records.append(record)
            results = pd.DataFrame.from_records(records)
            latex_output = results.to_latex(index=False, column_format='l' + 'c' * len(df.columns),
                                            float_format="{:0.3f}".format,
                                            caption=f"Results for {title.upper()} Dataset",label=f"tab:{title}_results")
            print(latex_output)