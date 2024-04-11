from torchrl.envs.utils import check_env_specs
from torchrl.envs import StepCounter, TransformedEnv

from src.utils import parse_args
from src.rl import *

import pickle
import shutil
import os
    
if __name__ == '__main__':
    args = parse_args()
    MODEL = args.model
    NUM_WORKERS = 4
    DATA_PATH = f'./data/'
    MODEL_SAVE_PATH = f'./models/{MODEL}/'

    shutil.rmtree(f'./logs/{MODEL}', ignore_errors=True)

    # load environment
    environment = SmartGlassesEnvironment(DATA_PATH)
    environment= TransformedEnv(environment, StepCounter())
    check_env_specs(environment)

    # train model
    model = eval(MODEL)(environment)
    model.train()

    # save model
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    with open(f'{MODEL_SAVE_PATH}/model.rl', 'wb') as handle:
        pickle.dump(model.policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
