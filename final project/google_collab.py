import pandas as pd
import numpy as np
import time
import torch
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

print("TensorFlow using GPU:", torch.cuda.is_available())
print("PyTorch using GPU:", torch.cuda.is_available())

print('--- Downloading NLTK Resources ---')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

from google.colab import files
uploaded = files.upload()

train_df = pd.read_json('banking77_train.json', lines=True)
test_df = pd.read_json('banking77_test.json', lines=True)

x_train = train_df['text'].tolist()
y_train = train_df['category'].tolist()
x_test = test_df['text'].tolist()
y_test = test_df['category'].tolist()

print('--- Clean and Preprocess Data ---')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

x_train = [clean_text(t) for t in x_train]
x_test = [clean_text(t) for t in x_test]

print('--- Loading Word2Vec Model ---')
word2vec_model = api.load('word2vec-google-news-300')

def get_w2v_embedding(text, model):
    words = text.split()
    word_vecs = [model[word] for word in words if word in model]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(model.vector_size)

x_train_word2vec = np.array([get_w2v_embedding(text, word2vec_model) for text in x_train])
x_test_word2vec = np.array([get_w2v_embedding(text, word2vec_model) for text in x_test])

print('--- GloVe Model ---')
glove_model = api.load("glove-wiki-gigaword-50")

def get_glove_embedding(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros((50))

x_train_glove = np.array([get_glove_embedding(text, glove_model) for text in x_train])
x_test_glove = np.array([get_glove_embedding(text, glove_model) for text in x_test])

print('--- Sentence-BERT Model ---')
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

def get_sbert_embedding(text, model):
    return model.encode(text)

x_train_sbert = np.array([get_sbert_embedding(text, sbert_model) for text in x_train])
x_test_sbert = np.array([get_sbert_embedding(text, sbert_model) for text in x_test])

print('--- Define Embeddings and Classifiers ---')
embeddings = {
    'Word2Vec': (x_train_word2vec, x_test_word2vec),
    'GloVe': (x_train_glove, x_test_glove), 
    'SBERT': (x_train_sbert, x_test_sbert)
}

classifier_params = {
    'SVM': (SVC(), {
        'C': [0.1, 1],
        'kernel': ['linear'],
        'gamma': ['scale']
    }),
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'min_samples_split': [2]
    }),
    'DecisionTree': (DecisionTreeClassifier(), {
        'max_depth': [None, 10],
        'min_samples_split': [2]
    })
}

print('--- Train, Tune, and Evaluate ---')
results = []

for embed_name, (X_tr, X_te) in embeddings.items():
    print(f"\nUsing Embedding: {embed_name}")
    for clf_name, (clf, param_grid) in classifier_params.items():
        print(f" Tuning and Training Classifier: {clf_name}")

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

results_df = pd.DataFrame(results)

# Save to TXT
with open("results.txt", "w") as f:
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    for res in results:
        f.write(f"Embedding: {res['Embedding']}\n")
        f.write(f"Classifier: {res['Classifier']}\n")
        f.write(f"Best Params: {res['Best Params']}\n")
        f.write(f"Accuracy: {res['Accuracy']:.4f}\n")
        f.write(f"F1 Score: {res['F1 Score']:.4f}\n")
        f.write(f"Training Time (s): {res['Training Time (s)']:.2f}\n")
        f.write(f"Testing Time (s): {res['Testing Time (s)']:.2f}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(res['Confusion Matrix'], max_line_width=np.inf))
        f.write("\n" + "="*80 + "\n\n")

# Download results
files.download("google_collab_results.txt")

# Show summary
print("\nFinal Results Summary:")
print(results_df[['Embedding', 'Classifier', 'Best Params', 'Accuracy', 'F1 Score', 'Training Time (s)', 'Testing Time (s)']])