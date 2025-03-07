import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidVectorizer
from sklearn.feature_selection import chi2, SelectKBest
import gensim.downloader as api

# Load Data
splits = {'train': 'split/train-00000-of-00001.parquet', 'validation': 'split/validation-00000-of-00001.parquet', 'test': 'split/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
train_text, train_label = df['text'].tolist(), df['label'].tolist()

df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])
test_text, test_label = df['text'].tolist(), df['label'].tolist()

print(train_text[:5])  # Print first 5 examples to verify

# Function to clean text (lowercase + remove punctuation)
def clean_text(text):
    text = text.lower()
    text = "".join(char for char in text if char.isalpha() or char.isspace())  # Keep only letters and spaces
    return text

# Apply cleaning
train_text = [clean_text(text) for text in train_text]
test_text = [clean_text(text) for text in test_text]

# Load stopwords
stop_words = set(stopwords.words('english')) 

# Remove stopwords with proper tokenization
train_text = [' '.join([word for word in word_tokenize(text) if word not in stop_words]) for text in train_text]
test_text = [' '.join([word for word in word_tokenize(text) if word not in stop_words]) for text in test_text]

print(train_text[:5])  # Print first 5 examples to verify

y_train, y_test = np.array(train_label), np.array(test_label)

# Bag of Words & TF-IDF 
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train_text)
x_test = vectorizer.transform(test_text)

tf_idf = TfidVectorizer()
x_train_tf_idf = tf_idf.fit_transform(train_text)
x_test_tf_idf = tf_idf.transform(test_text)

# Chi-Square Test
k_best = 100
selector = SelectKBest(chi2, k=k_best)
x_train = selector.fit_transform(x_train, y_train)
x_test = selector.transform(x_test)
x_train_tf_idf = selector.fit(x_train_tf_idf, y_train)
x_test_tf_idf = selector.transform(x_test_tf_idf)

# Glove Model
glove_model = api.load("glove-wiki-gigaword-50")

def get_embedding(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0, keepdims=True) if words_vectors else np.zeros((1, 50))

x_train_glove = np.array([get_embedding(text, glove_model) for text in train_text])
x_test_glove = np.array([get_embedding(text, glove_model) for text in test_text])
