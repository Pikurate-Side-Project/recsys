import argparse

import pickle
import pandas as pd 

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--train_fn', required=True)
    p.add_argument('--model_fn', required=True)
    p.add_argument('--pik_id', type=int, required=True)
    
    config = p.parse_args()

    return config

def main(config):
    with open(config.model_fn, 'rb') as m:
        loaded_model = pickle.load(m)

    df = pd.read_csv(config.train_fn).fillna('')
    sample = df.loc[df.pik_id == config.pik_id, :]

    print('\n', list(sample.pik_title)[0])

    try:
        results = loaded_model.predict(sample)
        for l in results:
            print('------------------%2d(%.2f)------------------' % (l['label'], l['ratio']))
            for d in l['contents']:
                print('%d\t%s' % (d[0], d[1]['title']))
    except Exception:
        print('픽 관련 자료가 없습니다.')

if __name__ == '__main__':
    config = define_argparser()
    main(config)