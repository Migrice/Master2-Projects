
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec, KeyedVectors


# define the function which takes as input the sentence and the model and return sentence embedding
def SentenceEmbedding(sentence, model, sentence_length):

    # preprocess the text
    preprocess_text = gensim.utils.simple_preprocess(sentence)
    word_to_add = "aklo"

    while(len(preprocess_text) < sentence_length):
        preprocess_text.insert(0, word_to_add)

    sentence_embedding = []
    for word in preprocess_text:
        try:
            sentence_embedding.append(model[word])
        except:
            sentence_embedding.append(np.zeros(300))

    return np.array(sentence_embedding)



# dataset embedding
def Dataset_embedding(data, model, max_seq_len):
    dataset_embedding = []
    for sentence in data:
        dataset_embedding.append(
            SentenceEmbedding(sentence, model, max_seq_len))

    return np.array(dataset_embedding)
