from src.utils import parse_args
from src.selector.llmselector import LLMSelector
from src.selector.randomselector import RandomSelector
from src.selector.idealselector import IdealSelector

if __name__ == '__main__':
    args = parse_args()
    MODEL = args.model
    DATA_PATH = f'./data/'
    MODEL_SAVE_PATH = f'./models/{MODEL}/'

    selector = LLMSelector(MODEL, MODEL_SAVE_PATH, DATA_PATH)
    print(selector.get_means())

    selector = RandomSelector(DATA_PATH)
    print(selector.get_means())

    selector = IdealSelector(DATA_PATH)
    print(selector.get_means())