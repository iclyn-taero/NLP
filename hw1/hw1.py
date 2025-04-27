import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
import gensim.downloader as api
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load Data
splits = {'train': 'split/train-00000-of-00001.parquet', 'validation': 'split/validation-00000-of-00001.parquet', 'test': 'split/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
train_text, train_label = df['text'].tolist(), df['label'].tolist()

df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])
test_text, test_label = df['text'].tolist(), df['label'].tolist()

# Print first 5 examples to verify
print(train_text[:5])

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

# Print first 5 examples to verify
print(train_text[:5])

y_train, y_test = np.array(train_label), np.array(test_label)

# Bag of Words & TF-IDF 
vectorizer = CountVectorizer()
x_train_bow = vectorizer.fit_transform(train_text)
x_test_bow = vectorizer.transform(test_text)

tf_idf = TfidfVectorizer()
x_train_tf_idf = tf_idf.fit_transform(train_text)
x_test_tf_idf = tf_idf.transform(test_text)

# Chi-Square Test
k_best = 100
selector = SelectKBest(chi2, k=k_best)
x_train_bow = selector.fit_transform(x_train_bow, y_train).toarray()  
x_test_bow = selector.transform(x_test_bow).toarray() 
x_train_tf_idf = selector.fit_transform(x_train_tf_idf, y_train).toarray()  
x_test_tf_idf = selector.transform(x_test_tf_idf).toarray()  

# Glove Model
glove_model = api.load("glove-wiki-gigaword-50")

def get_embedding(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0, keepdims=True) if word_vectors else np.zeros((50))

x_train_glove = np.array([get_embedding(text, glove_model) for text in train_text])
x_test_glove = np.array([get_embedding(text, glove_model) for text in test_text])
x_train_glove = x_train_glove.reshape(x_train_glove.shape[0], -1)
x_test_glove = x_test_glove.reshape(x_test_glove.shape[0], -1)

# Word2Vec Model 
word2vec = api.load("word2vec-google-news-300")
x_train_word2vec = np.array([get_embedding(text, word2vec) for text in train_text])
x_test_word2vec = np.array([get_embedding(text, word2vec) for text in test_text])
x_train_word2vec = x_train_word2vec.reshape(x_train_word2vec.shape[0], -1)
x_test_word2vec = x_test_word2vec.reshape(x_test_word2vec.shape[0], -1)

# USE Model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
x_train_use = use_model(train_text).numpy()
x_test_use = use_model(test_text).numpy()

# Function to train and evaluate classifiers
def train_and_evaluate(X_train, X_test, y_train, y_test, name):
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42),
        "XGBoost": XGBClassifier(eval_metric='mlogloss')
    }

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{clf_name} with {name}: Accuracy = {acc:.4f}")
        # Print out the misclassified examples
        #print(f"\nMisclassified Examples for {clf_name} with {name}:")
        #misclassified_idx = np.where(y_pred != y_test)[0]
        

        #for idx in misclassified_idx[:10]:
         #   print(f"Text: {test_text[idx]}")
         #   print(f"True Label: {y_test[idx]}")
         #   print(f"Predicted Label: {y_pred[idx]}")
         #   print("-" * 50)

# Train and evaluate models on different embeddings
train_and_evaluate(x_train_bow, x_test_bow, y_train, y_test, "Bag of Words")
train_and_evaluate(x_train_tf_idf, x_test_tf_idf, y_train, y_test, "TF-IDF")
train_and_evaluate(x_train_glove, x_test_glove, y_train, y_test, "GloVe")
train_and_evaluate(x_train_word2vec, x_test_word2vec, y_train, y_test, "Word2Vec")
train_and_evaluate(x_train_use, x_test_use, y_train, y_test, "USE")