import json
from collections import OrderedDict, defaultdict
import re
from keybert import KeyBERT

kw_model = KeyBERT()
dataPhrases = defaultdict()
contentData = json.load(open('content_data_new_meta.json','r',encoding='utf-8'))
for data in contentData:
    reviewList = []
    movieTitle = data['title'] + ' ' + data['year']
    for doc in data['reviews']:
        keywords  = kw_model.extract_keywords(doc,keyphrase_ngram_range=(1,10), stop_words=None, use_maxsum = True,top_n =5, use_mmr=True, diversity=0.5)
        reviewList.append(keywords)
    dataPhrases[movieTitle] = reviewList



with open('../../../experiment/keyBERT/230424_ngram10_diversity05.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(dataPhrases))
