import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class KmeansModel:

    def __init__(self, tokenizer, config):
        self._model = None
        self._labels= None
        
        self._index_to_pik = None
        self._pik_to_title = None
        self._pik_to_num = None

        self._tfidf_vectorizer_pik = None
        self._tfidf_vectorizer_category = None
        self._tfidf_vectorizer_link = None

        self.tokenzier = tokenizer
        self.n_cluster = config.n_cluster
        self.min_df = config.min_df
        self.max_feature = config.max_feature
    
    def train(self, training_data):
        
        unique_pik = dict(set(zip(training_data.pik_id, training_data.pik_title)))
        unique_category = dict(set(zip(training_data.category_id, training_data.category_title)))
        unique_link = dict(set(zip(training_data.link_id, training_data.memo)))

        index_to_pik = {idx: pik_id for idx, pik_id in enumerate(training_data.pik_id)}

        pik_to_index = {x: idx for idx, x in enumerate(list(unique_pik.keys()))}
        category_to_index = {x: idx for idx, x in enumerate(list(unique_category.keys()))}
        link_to_index = {x: idx for idx, x in enumerate(list(unique_link.keys()))}

        pik_to_num = training_data.groupby(['pik_id'], as_index=False).count()
        pik_to_num = {row[0]: row[1] for _, row in pik_to_num.iterrows()}

        tfidf_vectorizer_pik = TfidfVectorizer(tokenizer=self.tokenzier, \
            min_df=self.min_df, max_features=self.max_feature)
        tfidf_vectorizer_category = TfidfVectorizer(tokenizer=self.tokenzier, \
            min_df=self.min_df, max_features=self.max_feature)
        tfidf_vectorizer_link = TfidfVectorizer(tokenizer=self.tokenzier, \
            min_df=self.min_df, max_features=self.max_feature)
        
        # logging.info(f'train tfidf models, min_df: {self.min_df}, max_feature: {self.max_feature}')
        print(f'[INFO] train tfidf models[min_df: {self.min_df}, max_feature: {self.max_feature}]')

        vec_pik = tfidf_vectorizer_pik.fit_transform(list(unique_pik.values())).toarray()
        vec_category = tfidf_vectorizer_category.fit_transform(list(unique_category.values())).toarray()
        vec_link = tfidf_vectorizer_link.fit_transform(list(unique_link.values())).toarray()

        concated_matrix = KmeansModel.concat_matrix(training_data, vec_pik, vec_category, vec_link, pik_to_index, category_to_index, link_to_index)

        # logging.info(f'training model by using kmeans algorithm, n_cluster: {self.n_cluster}')
        print(f'[INFO] train model by using kmeans algorithm[n_cluster: {self.n_cluster}]')

        Kmeans = KMeans(n_clusters=self.n_cluster)
        Kmeans.fit(concated_matrix)
        
        # logging.info('finish!!')
        print('[INFO] finish!!')
        
        # allocate
        self._model = Kmeans
        self._labels= self._model.predict(concated_matrix)
        
        self._index_to_pik = index_to_pik
        self._pik_to_title = unique_pik
        self._pik_to_num = pik_to_num

        self._tfidf_vectorizer_pik = tfidf_vectorizer_pik
        self._tfidf_vectorizer_category = tfidf_vectorizer_category
        self._tfidf_vectorizer_link = tfidf_vectorizer_link
        

    def predict(self, sample):

        assert self._model, f'The model is not trained yet, you must train first!'
        assert self._tfidf_vectorizer_pik, f'The model is not trained yet, you must train first!'
        assert self._tfidf_vectorizer_category, f'The model is not trained yet, you must train first!'
        assert self._tfidf_vectorizer_link, f'The model is not trained yet, you must train first!'

        # make concat_matrix
        unique_pik = dict(set(zip(sample.pik_id, sample.pik_title)))
        unique_category = dict(set(zip(sample.category_id, sample.category_title)))
        unique_link = dict(set(zip(sample.link_id, sample.memo)))

        pik_to_index = {x: idx for idx, x in enumerate(list(unique_pik.keys()))}
        category_to_index = {x: idx for idx, x in enumerate(list(unique_category.keys()))}
        link_to_index = {x: idx for idx, x in enumerate(list(unique_link.keys()))}

        # extract label
        vec_pik = self._tfidf_vectorizer_pik.transform(list(unique_pik.values())).toarray()
        vec_category = self._tfidf_vectorizer_category.transform(list(unique_category.values())).toarray()
        vec_link = self._tfidf_vectorizer_link.transform(list(unique_link.values())).toarray()

        pik_matrix = KmeansModel.concat_matrix(sample, vec_pik, vec_category, vec_link, pik_to_index, category_to_index, link_to_index)
        
        # analysis
        labels = self._model.predict(pik_matrix)
        print(labels)
        link_num = len(labels)
        
        from collections import Counter
        label_freq = dict(Counter(labels))
        result = [{'label': label, 'ratio': num / link_num, 'contents': self.extract_statics_by_label(label)[:5] } for label, num in label_freq.items()]

        return result

    def extract_statics_by_label(self, target_label, desc=True):
        
        assert self._index_to_pik, f'The model is not trained yet, you must train first!'
        assert self._pik_to_title, f'The model is not trained yet, you must train first!'
        assert self._pik_to_num, f'The model is not trained yet, you must train first!'

        from collections import Counter
        
        target_piks = [self._index_to_pik[idx] for idx, label in enumerate(self._labels) 
            if label == target_label]
        pik_freq = dict(Counter(target_piks))
        result = {pik_id: {'title': self._pik_to_title[pik_id], 'ratio': num / self._pik_to_num[pik_id]} for pik_id, num in pik_freq.items()}
        
        if desc:
            result = sorted(result.items(), key=lambda x: x[1].get('ratio'))
            result.reverse()
        
        return result
        
    @staticmethod
    def concat_matrix(
        training_data,
        pik_matrix, category_matrix, link_matrix,
        pik_to_index, category_to_index, link_to_index
    ):
        pik_matrix_index = [pik_to_index[i] for i in training_data.pik_id]
        category_matrix_index = [category_to_index[i] for i in training_data.category_id]
        link_matrix_index = [link_to_index[i] for i in training_data.link_id]

        tmp_pik_vec = pik_matrix[pik_matrix_index]
        tmp_category_vec = category_matrix[category_matrix_index]
        tmp_link_vec = link_matrix[link_matrix_index]

        concat_matrix = np.concatenate((tmp_pik_vec, tmp_category_vec, tmp_link_vec), axis=1)

        return concat_matrix