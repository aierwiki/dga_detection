#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd



def get_malicious():
    import re
    records = []
    with open('./data/dga.txt') as f:
        records = re.findall(r'(\w+)\t+([\w.]+).*\n', f.read())
    df_malicious = pd.DataFrame({'Domain':[record[1] for record in records], 'Label':[record[0] for record in records]})
    return df_malicious



def get_benign():
    df_benign = pd.read_csv('./data/top-1m.csv', index_col = 0, header = None)
    df_benign.columns = ['Domain']
    df_benign['Label'] = 'benign'
    return df_benign



def prepare_data():
    import tldextract
    df_malicious = get_malicious()
    df_benign = get_benign()
    df_data = pd.concat([df_malicious, df_benign], axis = 0)
    df_data['Target'] = df_data['Label'].map(lambda x : 0 if x == 'benign' else 1)
    df_data['Domain'] = df_data['Domain'].map(lambda x : tldextract.extract(x).domain)
    return df_data


def make_data(df_data):
    df_data_small = pd.concat([df_data[df_data['Target'] == 0].sample(200000), df_data[df_data['Target'] == 1].sample(100000)], axis = 0)
    X = df_data_small['Domain'].values
    y = df_data_small['Target'].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2019, test_size = 0.2)
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(char_level = True)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_train = pad_sequences(X_train, padding = 'post')
    X_test = pad_sequences(X_test, padding = 'post', maxlen = X_train.shape[1])
    return X_train, X_test, y_train, y_test, tokenizer.word_index


# In[76]:



def build_model(words_num, max_length, feature_num):
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout, Activation
    model = Sequential()
    model.add(Embedding(input_dim = words_num, output_dim = feature_num, input_length = max_length))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    import keras.backend as K
    def calc_recall_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def calc_precision_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def calc_f1_score(y_true, y_pred):
        precision = calc_precision_score(y_true, y_pred)
        recall = calc_recall_score(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', 
                  metrics = ['acc', calc_recall_score, calc_precision_score, calc_f1_score])
    return model


# In[86]:


def train(X_train, X_test, y_train, y_test, word_index):
    model = build_model(len(word_index), X_train.shape[1], 128)
    model.fit(X_train, y_train, batch_size = 64, epochs = 5, validation_data = (X_test, y_test))


# In[ ]:
def main():
    df_data = prepare_data()
    X_train, X_test, y_train, y_test, word_index = make_data(df_data)
    train(X_train, X_test, y_train, y_test, word_index)


if __name__ == '__main__':
    main()


# In[ ]:




