import argparse
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Script parameters')

    # Add arguments
    parser.add_argument('--model', type=str, default='FCNNet', help='Model type')
    parser.add_argument('--num_trials', type=int, default=50, help='Number of optuna trials to run')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of samples in each mini-batch during neural network trainins')
    parser.add_argument('--cuda_device', default=0, type=int, help='Cuda device rank to train on')
    args = parser.parse_args()
    return args

def get_cuda_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'cpu'
    else:
        device = 'cpu'
    return device