# pip install pandas numpy gensim scikit-learn tensorflow tensorflow-hub sentence-transformers

import pandas as pd
import numpy as np
import time 
import tensorflow_hub as hub
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import gensim.downloader as api
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

import warnings
warnings.filterwarnings('ignore')

# --- 1. Download NLTK Resources ---
print('--- Downloading NLTK Resources ---')
# Ensure NLTK resources are only downloaded once
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# --- 2. Load Dataset ---
train_df = pd.read_json('banking77_train.json', lines=True)
test_df = pd.read_json('banking77_test.json', lines=True)

x_train = train_df['text'].tolist()
y_train = train_df['category'].tolist()

x_test = test_df['text'].tolist()
y_test = test_df['category'].tolist()

# --- 3. Clean and Preprocess Text ---
print('--- Clean and Preprocess Data ---')
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])  # Keep only letters and spaces
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

x_train = [clean_text(t) for t in x_train]
x_test = [clean_text(t) for t in x_test]

# --- 4. Load Embedding Models ---
print('--- Loading Word2Vec Model ---')
word2vec_model = api.load('word2vec-google-news-300')
# Function to get embeddings for a text
def get_w2v_embedding(text, model):
    words = text.split()
    word_vecs = [model[word] for word in words if word in model]
    if word_vecs:
        return np.mean(word_vecs, axis=0)
    else:
        return np.zeros(model.vector_size)  

x_train_word2vec = np.array([get_w2v_embedding(text, word2vec_model) for text in x_train])
x_test_word2vec = np.array([get_w2v_embedding(text, word2vec_model) for text in x_test])

x_train_word2vec = x_train_word2vec.reshape(x_train_word2vec.shape[0], -1)
x_test_word2vec = x_test_word2vec.reshape(x_test_word2vec.shape[0], -1)

# --- 5. Glove Model ---
print('--- GloVe Model ---')
glove_model = api.load("glove-wiki-gigaword-50")
x_train_glove, x_test_glove = [], []

def get_glove_embedding(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros((50``))

x_train_glove = np.array([get_glove_embedding(text, glove_model) for text in x_train])
x_test_glove = np.array([get_glove_embedding(text, glove_model) for text in x_test])

# --- 6. Use Model ---
print('--- Use Model ---')
use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
x_train_use = use_model(x_train).numpy()
x_test_use = use_model(x_test).numpy()

# --- 7. Sentence-Bert Model ---
print('--- Sentence-Bert Model ---')
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_sbert_embedding(text, model):
    return model.encode(text)

x_train_sbert = np.array([get_sbert_embedding(text, sbert_model) for text in x_train])
x_test_sbert = np.array([get_sbert_embedding(text, sbert_model) for text in x_test])

# --- 8. Define Embeddings and Classifiers ---
print(' --- Define Embeddings and Classifiers ---')
embeddings = {
    'Word2Vec': (x_train_word2vec, x_test_word2vec),
    'GloVe': (x_train_glove, x_test_glove), 
    'USE' : (x_train_use, x_test_use),
    'SBERT': (x_train_sbert, x_test_sbert)
}

classifier_params = {
    'SVM': (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }),
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }),
    'DecisionTree': (DecisionTreeClassifier(), {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    })
}

# ---- 9. Train, Tune, and Evaluate ----
print(' --- Train, Tune, and Evaluate ---')
results = []

for embed_name, (X_tr, X_te) in embeddings.items():
    print(f"\nUsing Embedding: {embed_name}")
    for clf_name, (clf, param_grid) in classifier_params.items():
        print(f" Tuning and Training Classifier: {clf_name}")

        # Hyperparameter search
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)

        start_train = time.time()
        grid_search.fit(X_tr, y_train)
        end_train = time.time()

        best_clf = grid_search.best_estimator_

        start_test = time.time()
        y_pred = best_clf.predict(X_te)
        end_test = time.time()

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'Embedding': embed_name,
            'Classifier': clf_name,
            'Best Params': grid_search.best_params_,
            'Accuracy': acc,
            'F1 Score': f1,
            'Training Time (s)': end_train - start_train,
            'Testing Time (s)': end_test - start_test,
            'Confusion Matrix': cm
        })

# ---- 9. Save and Print Results ----
results_df = pd.DataFrame(results)
print("\nFinal Results Summary:")
print(results_df[['Embedding', 'Classifier', 'Best Params', 'Accuracy', 'F1 Score', 'Training Time (s)', 'Testing Time (s)']])

results_df.to_csv('banking77_final_results.csv', index=False)