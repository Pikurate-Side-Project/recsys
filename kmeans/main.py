import sys
import argparse

import pickle
import pandas as pd

from model import KmeansModel
from utils import extract_noun


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--train_fn', required=True)
    p.add_argument('--model_fn', required=True)
    p.add_argument('--n_cluster', type=int, default=20)
    p.add_argument('--min_df', type=int, default=2)
    p.add_argument('--max_feature', type=int, default=500)
    
    config = p.parse_args()

    return config

def main(config):

    training_data = pd.read_csv(config.train_fn).fillna('')

    model = KmeansModel(extract_noun, config)
    model.train(training_data)

    with open(config.model_fn, 'wb') as m:
        pickle.dump(model, m)


if __name__ == '__main__':
    config = define_argparser()
    main(config)