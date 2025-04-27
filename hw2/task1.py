import re
import json
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import gensim
import spacy
import requests
from collections import Counter
from io import StringIO
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import CoherenceModel
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel

# Ensure NLTK resources are only downloaded once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess(text):
    text = re.sub(r"(writes:|says:|wrote:|>+)", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words and len(t) > 2]

if __name__ == "__main__":
    # Load training and test sets
    with open("train.json", "r") as f:
        train_data = json.load(f)
    with open("test.json", "r") as f:
        test_data = json.load(f)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # ================================
    # TASK 1: Topic Modeling with LDA
    # ================================
    print("\n🔹 TASK 1: Topic Modeling")

    # Preprocess the text
    preprocessed_docs = train_df['content'].map(preprocess)
    
    # Create dictionary and corpus for topic modeling
    dictionary = corpora.Dictionary(preprocessed_docs)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_docs]

    coherence_scores = []
    models = []

    for k in range(1, 11):
        print(f"Training LDA model for {k} topics...")
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary,
                                           num_topics=k, random_state=42, passes=10)
        coherence_model = CoherenceModel(model=lda_model, texts=preprocessed_docs,
                                         dictionary=dictionary, coherence='c_v')
        score = coherence_model.get_coherence()
        coherence_scores.append(score)
        models.append(lda_model)
        print(f"Coherence for {k} topics: {score:.4f}")

    # Plot coherence scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), coherence_scores, marker='o')
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Topics (K)")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Scores for Different Topic Counts")
    plt.grid(True)
    plt.show()

    # Best model
    best_k = coherence_scores.index(max(coherence_scores)) + 1
    best_model = models[best_k - 1]
    print(f"Best number of topics: {best_k} with coherence {max(coherence_scores):.4f}")

    # Generate word clouds
    for i in range(best_k):
        words = dict(best_model.show_topic(i, 30))
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {i + 1}")
        plt.show()
