import pickle
import pandas as pd 

from model import KmeansModel

if __name__ == '__main__':
    with open('./models/c15.d1500.pkl', 'rb') as m:
        loaded_model: KmeansModel = pickle.load(m)

    df = pd.read_csv('./data/link_data.csv').fillna('')
    sample = df.loc[df.pik_id == 5200, :]
    print(sample)
    # print(sample)
    # print(loaded_model.predict(sample))
    print(loaded_model.extract_statics_by_label(2))