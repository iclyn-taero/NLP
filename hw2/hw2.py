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

    # =============================================
    # TASK 2: NER + Affective Analysis using spaCy
    # =============================================
    print("\n🔹 TASK 2: NER and Affective Analysis")

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 15000000 
    full_text = " ".join(train_df['content'].tolist())
    doc = nlp(full_text)

    # Top 3 most frequent entity types
    entity_labels = [ent.label_ for ent in doc.ents]
    top_entities = Counter(entity_labels).most_common(3)
    print("Top 3 entity types:", top_entities)

    # Load NRC VAD Lexicon
    url = "https://saifmohammad.com/WebDocs/VAD/NRC-VAD-Lexicon.txt"
    vad_txt = requests.get(url).text
    vad_df = pd.read_csv(StringIO(vad_txt), sep='\t')
    vad_dict = dict(zip(vad_df['Word'], vad_df['Valence']))

    # Sentiment analysis
    results = {}
    for entity_type, _ in top_entities:
        sentiments = []
        for sent in doc.sents:
            for token in sent:
                if token.ent_type_ == entity_type and token.dep_ == "nsubj":
                    related = [
                        child.text.lower()
                        for child in token.children
                        if child.pos_ in ["VERB", "ADJ"]
                    ]
                    sentiments += [vad_dict[w] for w in related if w in vad_dict]
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
        results[entity_type] = avg_sent

    print("Average Valence Scores for Top Entities:")
    for ent, score in results.items():
        print(f" - {ent}: {score:.3f}")

    # =========================================
    # TASK 3: Classification with BERT (SimpleTransformers)
    # =========================================
    print("\n🔹 TASK 3: Text Classification using BERT")

    train_data = train_df[["content", "Newsgroup"]].copy()
    test_data = test_df[["content", "Newsgroup"]].copy()

    # Encode labels
    le = LabelEncoder()
    train_data["Newsgroup"] = le.fit_transform(train_data["Newsgroup"])
    test_data["Newsgroup"] = le.transform(test_data["Newsgroup"])
    train_data.columns = ["text", "labels"]
    test_data.columns = ["text", "labels"]

    # Initialize and train model
    model = ClassificationModel(
        "bert", "bert-base-uncased",
        num_labels=len(le.classes_),
        use_cuda=False,
        args={
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "num_train_epochs": 2,
            "save_model_every_epoch": False,
            "output_dir": "outputs/",
        }
    )

    model.train_model(train_data)

    # Evaluate
    predictions, _ = model.predict(test_data["text"].tolist())
    print("\nClassification Report:\n")
    print(classification_report(test_data["labels"], predictions, target_names=le.classes_))