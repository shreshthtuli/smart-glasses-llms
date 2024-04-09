from ..llmnet import *
from ..dataset import CTCDataset
from glob import glob
import pandas as pd
import numpy as np

class TemplateSelector():
    def __init__(self, DATA_PATH):
        self.data_path = DATA_PATH
        self.load_data()

    def load_data(self):
        # load answers
        all_answers = {}
        for bench_dir in glob("data/*/"):
            bench_name = bench_dir.split('/')[1]
            all_answers[bench_name] = {}
            for llm_path in glob(bench_dir+'model_answer/*.jsonl'):
                llm_name = llm_path.split('/')[3].split('.jsonl')[0]
                all_answers[bench_name][llm_name] = pd.read_json(llm_path, lines=True)
        all_bench_names = list(all_answers.keys())
        all_llm_names = list(all_answers[all_bench_names[0]].keys())
        # load judgements
        all_judgements = {}
        for bench_dir in glob("data/*/"):
            judge_path = glob(bench_dir+'model_judgment/*.jsonl')[0]
            df = pd.read_json(judge_path, lines=True)
            bench_name = bench_dir.split('/')[1]
            all_judgements[bench_name] = {}
            for llm_name in all_llm_names:
                all_judgements[bench_name][llm_name] = df[df.model == llm_name]
        # load train/test sets
        train_dset = pd.read_json(f'{self.data_path}/train.jsonl', lines=True)
        test_dset = pd.read_json(f'{self.data_path}/test.jsonl', lines=True)
        # combine datasets
        for dset in [train_dset, test_dset]:
            for llm_name in all_llm_names:
                col_time = []; col_score = []
                for _, row in dset.iterrows():
                    bench_name = row['bench']
                    ans = all_answers[bench_name][llm_name]
                    ans = ans[ans['question_id'] == row['question_id']].reset_index()
                    proc_time = np.mean([a['proc_time'] for a in ans.iloc[0]['choices']])
                    col_time.append(proc_time)
                    sco = all_judgements[bench_name][llm_name]
                    sco = sco[sco['question_id'] == row['question_id']].reset_index()
                    col_score.append(sco.iloc[0]['score'] if sco.iloc[0]['score'] > 0 else 0)
                dset.insert(len(dset.columns), llm_name+'_time', col_time)
                dset.insert(len(dset.columns), llm_name+'_score', col_score)
        self.all_bench_names = all_bench_names
        self.all_llm_names = all_llm_names
        self.train_dset = train_dset
        self.test_dset = test_dset

    def get_means(self):
        return self.test_dset.score.mean(), self.test_dset.time.mean(), self.test_dset.qos.mean(), self.test_dset.selection_time.mean()
        

