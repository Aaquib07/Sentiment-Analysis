import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from copy import deepcopy
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report


train_df = pd.read_csv('dataset/Twitter Sentiment Dataset/train.csv')

# In this datasets, there are 29720 negative tweets and 2242 positive tweets.
# print(train_df['label'].value_counts())

# A function add_to_dict has been defined that takes as input a file containing 
# vectors of different words(here glove embedding is used) and a dictionary and 
# populates the dictionary with the words as keys and their corresponding vectors as values 
words = dict()

def add_to_dict(d, filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split()
            
            try:
                d[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue


add_to_dict(words, 'glove/glove.6B.50d.txt')

tokenizer = nltk.RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

# A function message_to_token_list has been defined that takes as input a sentence 
# or tweet and tokenize it and converts all the tokens in that sentence to lowercase 
# and lemmatize all those tokens to their root words and finally returns a list of 
# root word tokens in that sentence. 
def message_to_token_list(s):
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words]
    
    return useful_tokens


# A function message_to_word_vectors has been defined that takes as input a sentence or 
# tweet and converts it to vector form and returns the final vectorized form.
def message_to_word_vectors(message, word_dict=words):
    processed_list_of_tokens = message_to_token_list(message)
    
    vectors = []
    
    for token in processed_list_of_tokens:
        if token not in word_dict:
            continue
        
        token_vector = word_dict[token]
        vectors.append(token_vector)
    
    return np.array(vectors, dtype=float)

# Now the dataset has been split into training set, validation set and testing set
train_df = train_df.sample(frac=1, random_state=1)
train_df.reset_index(drop=True, inplace=True)

split_index_1 = int(len(train_df) * 0.7)
split_index_2 = int(len(train_df) * 0.85)

train_df, val_df, test_df = train_df[:split_index_1], train_df[split_index_1:split_index_2], train_df[split_index_2:]

# len(train_df), len(val_df), len(test_df)


# A function df_to_X_y has been defined that takes as input a dataframe and separates 
# the dataframe into X (containing vectorized tweets) and y(containing labels of tweets)
def df_to_X_y(dff):
    y = dff['label'].to_numpy().astype(int)
    
    all_word_vector_sequences = []
    
    for message in dff['tweet']:
        message_as_vector_seq = message_to_word_vectors(message)
        
        if message_as_vector_seq.shape[0] == 0:
            message_as_vector_seq = np.zeros(shape=(1, 50))
            
        all_word_vector_sequences.append(message_as_vector_seq)
    
    return all_word_vector_sequences, y


X_train, y_train = df_to_X_y(train_df)

# Plotted a histogram that visualizes the sequence lengths of all tweets.
sequence_lengths = []

for i in range(len(X_train)):
    sequence_lengths.append(len(X_train[i]))
    
plt.hist(sequence_lengths)

# Mathematical statistics of sequence lengths
# print(pd.Series(sequence_lengths).describe())


# A function pad_X has been defined that takes as input a vectorized tweets dataframe 
# and a desired sequence length and returns a padded vectorized form so that every 
# input have the same dimension (i.e, 3-D).
def pad_X(X, desired_sequence_length=57):
    X_copy = deepcopy(X)
    
    for i, x in enumerate(X):
        x_seq_len = x.shape[0]
        sequence_length_difference = desired_sequence_length - x_seq_len
        
        pad = np.zeros(shape=(sequence_length_difference, 50))
        
        X_copy[i] = np.concatenate([x, pad])
        
    return np.array(X_copy).astype(float)


X_train = pad_X(X_train)

X_val, y_val = df_to_X_y(val_df)
X_val = pad_X(X_val)

X_test, y_test = df_to_X_y(test_df)
X_test = pad_X(X_test)


# A sequential tensorflow model have been built that consist of an Input layer followed 
# by 3 pairs of LSTM and Dropout layers, then a Flatten layer that converts all the 
# outputs originally in 2-D form to 1-D and lastly a Dense layer to get the output.
model = Sequential([])
model.add(layers.Input(shape=(57, 50)))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.Dropout(0.3))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.Dropout(0.3))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

cp = ModelCheckpoint('model/', save_best_only=True)

model.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(),
              metrics=['accuracy', AUC(name='auc')])

weights = {0: frequencies.sum() / frequencies[0],
           1: frequencies.sum() / frequencies[1]}

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20,
          callbacks=[cp], class_weight=weights)

best_model = load_model('model/')

test_predictions = (best_model.predict(X_test) > 0.5).astype(int)
# classification analysis
print(classification_report(y_test, test_predictions))