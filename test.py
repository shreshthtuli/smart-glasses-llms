from src.utils import parse_args
from src.selector.rlselector import RLSelector
from src.selector.llmselector import LLMSelector
from src.selector.randomselector import RandomSelector
from src.selector.idealselector import IdealSelector
from src.plotter import Plotter

import pandas as pd
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    args = parse_args()
    DATA_PATH = f'./data/'
    MODEL_SAVE_PATH = f'./models/'

    plotter = Plotter()

    selector = RandomSelector(DATA_PATH)
    plotter.add_results('Random', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = RLSelector('DDPGPolicy', MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('DDPG', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = RLSelector('PPOPolicy', MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('PPO', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = RLSelector('DQNPolicy', MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('DQN', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = RLSelector('SACPolicy', MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('SAC', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = LLMSelector('BranchyNet', MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('Branchy', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = LLMSelector('ZTWNet', MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('ZTW', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = LLMSelector('EENet', MODEL_SAVE_PATH, DATA_PATH, 1)
    plotter.add_results('EE1', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = LLMSelector('FCNNet', MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('FCN', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = LLMSelector('EENet', MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('EE', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    # selector = IdealSelector(DATA_PATH)
    # plotter.add_results('Oracle', selector.train_dset, selector.test_dset)
    # print(selector.get_means())

    plotter.plot_scores_times()
    plotter.gen_scores_times_table()