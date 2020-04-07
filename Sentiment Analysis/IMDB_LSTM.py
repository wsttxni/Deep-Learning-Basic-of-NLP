import os
import time
import copy
import keras
import gensim
import numpy as np
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Input,Dense,GRU,LSTM,Activation,Dropout,Embedding
from keras.layers import Multiply,Concatenate,Dot
np.random.seed(1)

pos_files = os.listdir('train/pos')
neg_files = os.listdir('tain/neg')

pos_all = []
neg_all = []
for pf, nf in zip(pos_files, neg_files):
    with open('train/pos'+'/'+pf, encoding='utf-8') as f:
        s = f.read()
        pos_all.append(s)
    with open('train/neg'+'/'+nf, encoding='utf-8') as f:
        s = f.read()
        neg_all.append(s)
# print(len(pos_all))
# print(len(neg_all))
X_orig = np.array(pos_all + neg_all)
Y_orig = np.array([1 for _ in range(12500)] + [0 for _ in range(12500)])
# print("X_orig:",X_orig.shape)
# print("Y_orig:",Y_orig.shape)

vocab_size = 20000
maxlen = 200
t = Tokenizer(vocab_size)
tik = time.time()
t.fit_on_texts(X_orig)
tok = time.time()
word_index = t.word_index
# print("Fitting time: ",(tok-tik),'s')
v_X = t.texts_to_sequences(X_orig)
pad_X = pad_sequences(v_X, maxlen = maxlen, padding = 'post')

x = list(t.word_counts.items())
x_sort = sorted(x, key = lambda p:p[1], reverse = True)
small_word_index = copy.deepcopy(word_index)
for item in x_sort[20000:]:
    small_word_index.pop(item[0])
# print(len(small_word_index))
# print(len(word_index))

model_file = 'data/GoogleNews-vectors-negative300.bin'
wv_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary = True)

embedding_matrix = np.random.uniform(size = (vocab_size+1, 300))
for word, index in small_word_index.items():
    try:
        word_vector = wv_model[word]
        embedding_matrix[index] = word_vector
    except:
        print("Word: [",word,"] not in wvmodel! Use random embedding instead.")
# print("Embedding matrix shape:\n", embedding_matrix.shape)

random_indexs = np.random.permutation(len(pad_X))
X = pad_X[random_indexs]
Y = Y_orig[random_indexs]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)
# print("X_test:", X_test.shape)
# print("y_test:", y_test.shape)

inputs = Input(shape = (maxlen,))
use_pretrained_wv = True
if use_pretrained_wv:
    word_vector = Embedding(VOCAB_SIZE+1, wv_dim, input_length=MAXLEN, weights=[embedding_matrix])(inputs)
else:
    word_vector = Embedding(VOCAB_SIZE+1, wv_dim, input_length=MAXLEN)(inputs)

hidden_layer = LSTM(128)(word_vector)
y_hat = Dense(1, activation = 'sigmoid')(hidden_layer)
model = Model(input = inputs, output = y_hat)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=5, validation_split=0.15)