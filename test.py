from src.utils import parse_args
from src.selector.llmselector import LLMSelector
from src.selector.randomselector import RandomSelector
from src.selector.idealselector import IdealSelector
from src.plotter import Plotter

if __name__ == '__main__':
    args = parse_args()
    MODEL = args.model
    DATA_PATH = f'./data/'
    MODEL_SAVE_PATH = f'./models/{MODEL}/'

    plotter = Plotter()

    selector = LLMSelector(MODEL, MODEL_SAVE_PATH, DATA_PATH)
    plotter.add_results('LLM', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = RandomSelector(DATA_PATH)
    plotter.add_results('Random', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    selector = IdealSelector(DATA_PATH)
    plotter.add_results('Ideal', selector.train_dset, selector.test_dset)
    print(selector.get_means())

    plotter.plot_scores_times()