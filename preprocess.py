import numpy as np
import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

#stopwords_list=pickle.load(open(os.path.join('pickleObjects','stopwords.pkl'),'rb'))

def preprocess(text):
    text=re.sub('<[^>]*>','',text) ## Removes HTML Markups
    text=re.sub('\n','',text) ## Removes newline
    text=re.sub(' +',' ',text) ## Removes multiple spaces between words
    text=''.join([x for x in text if x not in ['!','&']])
    text=[x.lower() for x in text.split()]
    return ' '.join(text)



    