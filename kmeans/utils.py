import re
from typing import List, Tuple

import nltk
from nltk.stem import WordNetLemmatizer
from konlpy.tag import Mecab

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

mecab = Mecab()
stemmer = WordNetLemmatizer()

def preprocess(text: str):
    
    result = re.sub(pattern=r'[\[\]():|]', repl='', string=text)
    result = re.sub(pattern=r'\s', repl=' ', string=result)
    result = re.sub(pattern=r'[一-龥]', repl='', string=result)
    result = re.sub(pattern=r'[ㄱ-ㅎㅏ-ㅣ]', repl=' ', string=result)

    result.strip()
    result = ' '.join(result.split())

    return result

def extract_noun(text: str, target_pos=['NNG', 'NNP', 'NNB']) -> List[str]:

    def is_noun(word: Tuple[str, str]) -> bool:

        if word[1] in target_pos:
            return True
        elif word[1] == 'SL' and nltk.pos_tag([word[0]])[0][1] == 'NN':
            return True
        return False

    pos_list : List[Tuple[str, str]] = mecab.pos(preprocess(text))
    result = [stemmer.lemmatize(p[0]) for p in pos_list if is_noun(p)]

    return result