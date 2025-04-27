import spacy
import pandas as pd
from io import StringIO
from collections import Counter
import os
from nltk.sentiment import SentimentIntensityAnalyzer

def run_task2(train_df):
    print("\n🔹 TASK 2: NER and Affective Analysis")

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 15000000

    # Initialize SentimentIntensityAnalyzer for VADER
    sid = SentimentIntensityAnalyzer()

    # Path to local NRC VAD Lexicon file
    lexicon_path = '/Users/icemac/nltk_data/sentiment/vader_lexicon/vader_lexicon.txt'
    
    try:
        # Load the NRC VAD Lexicon from the local file
        with open(lexicon_path, 'r') as file:
            vad_txt = file.read()

        # Read the lexicon into a DataFrame
        vad_df = pd.read_csv(StringIO(vad_txt), sep='\t')
        
        # Print the first few rows and column names to debug
        print("Columns in NRC VAD Lexicon:", vad_df.columns)
        print("First few rows of NRC VAD Lexicon:", vad_df.head())

        # Check if the NRC VAD Lexicon contains the expected structure
        if len(vad_df.columns) >= 3:
            # Assuming first column contains the words, and the second column contains valence scores
            vad_dict = dict(zip(vad_df.iloc[:, 0], vad_df.iloc[:, 1]))  # Adjust as needed
        else:
            print("Error: NRC VAD Lexicon doesn't have expected columns.")
            return

        entity_counter = Counter()
        valence_scores = {}
        sentiment_results = []

        for text in train_df['content']:
            doc = nlp(text)

            # Entity extraction using Spacy NER
            for ent in doc.ents:
                entity_counter[ent.label_] += 1

            # Sentence-based valence score calculation
            for sent in doc.sents:
                for token in sent:
                    ent_type = token.ent_type_
                    if ent_type and token.dep_ == "nsubj":
                        related = [
                            child.text.lower()
                            for child in token.children
                            if child.pos_ in ["VERB", "ADJ"]
                        ]
                        scores = [vad_dict[w] for w in related if w in vad_dict]
                        if scores:
                            valence_scores.setdefault(ent_type, []).extend(scores)

            # VADER sentiment analysis for valence scoring
            sentiment = sid.polarity_scores(text)
            sentiment_results.append(sentiment)

        # After processing all texts, print summarized results
        top_entities = entity_counter.most_common(3)
        print("Top 3 entity types:", top_entities)

        print("Average Valence Scores for Top Entities:")
        for ent_type, _ in top_entities:
            scores = valence_scores.get(ent_type, [])
            avg = sum(scores) / len(scores) if scores else 0
            print(f" - {ent_type}: {avg:.3f}")

        # Example of summarizing sentiment results
        avg_sentiment = {
            "neg": sum([sent["neg"] for sent in sentiment_results]) / len(sentiment_results),
            "neu": sum([sent["neu"] for sent in sentiment_results]) / len(sentiment_results),
            "pos": sum([sent["pos"] for sent in sentiment_results]) / len(sentiment_results),
            "compound": sum([sent["compound"] for sent in sentiment_results]) / len(sentiment_results),
        }

        print(f"\nAverage Sentiment Scores:\n{avg_sentiment}")

    except Exception as e:
        print("Error reading the NRC VAD Lexicon file:", e)


if __name__ == "__main__":
    # Get the path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct paths for train and test JSON files (in the same folder)
    train_df_path = os.path.join(script_dir, "train.json")
    test_df_path = os.path.join(script_dir, "test.json")

    try:
        # Load the training and testing data
        train_df = pd.read_json(train_df_path)
        test_df = pd.read_json(test_df_path)

        print(f"Successfully loaded training data from {train_df_path}")
        run_task2(train_df)

    except Exception as e:
        print("Failed to load or process the dataset:", e)
# Successfully loaded training data from /Users/icemac/Documents/School/Masters/NLP/hw2/train.json

# Successfully loaded training data from /Users/icemac/Documents/School/Masters/NLP/hw2/train.json
# This means the script successfully loaded the training dataset from the specified file path.
#
# 🔹 TASK 2: NER and Affective Analysis
# This marks the beginning of the second task, which is about Named Entity Recognition (NER) and 
# Affective Analysis (emotion/valence analysis) of the text in the training dataset.

# Columns in NRC VAD Lexicon: Index(['$:', '-1.5', '0.80623', '[-1, -1, -1, -1, -3, -1, -3, -1, -2, -1]'], dtype='object')
# This shows the columns of the NRC VAD Lexicon, where the format might not be ideal for use, as 
# the lexicon's columns seem to have irregular headers, not matching the expected "Word" and "Valence".

# First few rows of NRC VAD Lexicon: 
#          $:  -1.5  0.80623 [-1, -1, -1, -1, -3, -1, -3, -1, -2, -1]
# 0        %)  -0.4  1.01980      [-1, 0, -1, 0, 0, -2, -1, 2, -1, 0]
# 1       %-)  -1.5  1.43178   [-2, 0, -2, -2, -1, 2, -2, -3, -2, -3]
# 2       &-:  -0.4  1.42829     [-3, -1, 0, 0, -1, -1, -1, 2, -1, 2]
# 3        &:  -0.7  0.64031   [0, -1, -1, -1, 1, -1, -1, -1, -1, -1]
# 4  ( '}{' )   1.6  0.66332           [1, 2, 2, 1, 1, 2, 2, 1, 3, 1]
# This output shows some of the first few rows in the NRC VAD Lexicon, which contains symbolic representations
# with values that may represent valence scores (emotional values), though the format seems a bit off from the 
# expected structure. The lexicon likely needs cleaning for further analysis.

# Top 3 entity types: [('ORG', 28561), ('PERSON', 28272), ('CARDINAL', 20314)]
# This indicates that the most frequently identified entity types in the training data were:
# - ORG (Organization), with 28,561 occurrences
# - PERSON, with 28,272 occurrences
# - CARDINAL (numbers or quantities), with 20,314 occurrences
# This suggests that the text contains a lot of references to organizations, people, and numerical values, 
# which are key entities for many NLP tasks like named entity recognition.

# Average Valence Scores for Top Entities:
#  - ORG: 1.033
#  - PERSON: 0.347
#  - CARDINAL: -0.625
# This shows the average valence scores (emotional sentiment scores) for the top entities:
# - ORG (Organizations) have an average valence score of 1.033, indicating positive sentiment associated with 
#   organizations in the text.
# - PERSON (People) have an average valence score of 0.347, which is a relatively neutral sentiment.
# - CARDINAL (Numerical values) have an average valence score of -0.625, suggesting that numbers or quantities 
#   tend to be associated with slightly negative sentiment.

# Average Sentiment Scores:
# {'neg': 0.07039811089732367, 'neu': 0.8365618331292644, 'pos': 0.0930395312226692, 'compound': 0.14578329543466848}
# These are the overall sentiment scores derived from VADER sentiment analysis:
# - Negative (neg): 0.0704 (The text has a very low negative sentiment overall)
# - Neutral (neu): 0.8366 (The majority of the text is neutral)
# - Positive (pos): 0.0930 (The positive sentiment is also low)
# - Compound: 0.1458 (This is a composite score combining the other values, showing a slight positive sentiment in 
#   the text overall. Positive values close to 1.0 indicate positive sentiment, values close to -1.0 indicate 
#   negative sentiment, and values near 0 indicate neutrality.)
