import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

# Reference: https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html

moviereview_train = pd.read_csv('train.tsv', delimiter='\t')
moviereview_test = pd.read_csv('test.tsv', delimiter='\t')
print(moviereview_train.info())

# check for null
print(moviereview_train.isnull().sum())
print(moviereview_test.isnull().sum())
# no null values

# check for duplicates
print(moviereview_train[moviereview_train.duplicated(['PhraseId'], keep=False)])
print(moviereview_test[moviereview_test.duplicated(['PhraseId'], keep=False)])
# no duplicates

print(moviereview_train.head(10))

# Data analysis
# count plot
sns.countplot(x= 'Sentiment', data= moviereview_train)
plt.show()

# Labels are as follows:
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive
# Neutral ranks the highest

# Dropping PhraseId and SentenceID
moviereview_train.drop(['PhraseId', 'SentenceId'], inplace=True, axis=1)
moviereview_test.drop(['PhraseId', 'SentenceId'], inplace=True, axis=1)

# hyperparameters
num_words = 5000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

# Preprocessing with TensorFlow and Keras
# Tokenize training data
tokenizer = Tokenizer(num_words = num_words, oov_token= oov_token)
tokenizer.fit_on_texts(moviereview_train['Phrase'])
# Get training data word index
word_index = tokenizer.word_index
# Encode training data sentences into sequences
train_sequences = tokenizer.texts_to_sequences(moviereview_train['Phrase'])
# Get max training sequence length
maxlen = max([len(x) for x in train_sequences])
# Pad the training sequences
train_padded = pad_sequences(train_sequences,
                             padding = pad_type,
                             truncating= trunc_type,
                             maxlen= maxlen)

# Repeat for test data

tokenizer.fit_on_texts(moviereview_test['Phrase'])
test_sequences = tokenizer.texts_to_sequences(moviereview_test['Phrase'])
test_padded = pad_sequences(test_sequences,
                            padding = pad_type,
                            truncating=trunc_type,
                            maxlen= maxlen)

# train test split
X = train_padded
# Label
Y = pd.get_dummies(moviereview_train['Sentiment'])
X_train, X_lab, Y_train, Y_lab = train_test_split(X, Y,
                                                 test_size=0.3)


embedding_vector_length = 32

# Reference:  https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37#:~:text=Building%20the%20model&text=Sequential%20is%20the%20easiest%20way,add%20layers%20to%20our%20model.

# Creating model
model = Sequential()
# get number of columns in training data
n_cols = X_train.shape[1]
# add model layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dense(5,activation = 'softmax'))

# model compilation
model.compile(loss = 'categorical_crossentropy',
                        optimizer = 'adam',
                        metrics=['accuracy'])
print(model.summary())
train_history = model.fit(x = X_train,
                          y = Y_train,
                          batch_size= 64,
                          epochs=10,
                          verbose=2,
                          validation_data=(X_lab, Y_lab))

# Plot accuracy
plt.figure(figsize=(10,10))
# Plotting accuracy of training set
plt.plot(train_history.history['accuracy'], 'r', label= 'Training Accuracy')
# plotting accuracy of validation set
plt.plot(train_history.history['val_accuracy'], 'b', label= 'Training Accuracy')
plt.legend()
plt.show()










