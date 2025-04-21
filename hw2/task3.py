import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def run_task3(train_df):
    print("\n🔹 TASK 3: Text Classification")

    # Rename 'Newsgroup' to 'label'
    train_df = train_df.rename(columns={'Newsgroup': 'label'})

    # Fill missing content
    train_df['content'] = train_df['content'].fillna("")

    # TF-IDF Vectorization
    print("✅ Extracting features with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(train_df['content'])
    y = train_df['label']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Classifier
    print("✅ Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    try:
        # Load training data from a JSON file
        train_df = pd.read_json("train.json")  # Update path if needed

        # Rename 'Newsgroup' to 'label' to match expected column names
        if 'Newsgroup' in train_df.columns:
            train_df.rename(columns={'Newsgroup': 'label'}, inplace=True)

        # Print out columns and the first few rows to inspect the dataset structure
        print("Columns in the dataset:", train_df.columns)
        print("\nFirst few rows of the dataset:")
        print(train_df.head())

        # Check if the necessary columns are present
        if 'label' not in train_df.columns or 'content' not in train_df.columns:
            print("❌ 'label' or 'content' column not found in dataset.")
        else:
            # Proceed with the classification task
            run_task3(train_df)

    except Exception as e:
        print("❌ Error loading data or running Task 3:", e)

# 🔹 TASK 3: Text Classification
# ✅ Extracting features with TF-IDF...
# ✅ Training Logistic Regression...

# 📊 Classification Report:
#                          precision    recall  f1-score   support
#
# comp.os.ms-windows.misc       0.93      1.00      0.96       269
#         rec.motorcycles       0.94      0.98      0.96       182
#  soc.religion.christian       0.96      0.96      0.96       235
#      talk.politics.guns       0.93      0.92      0.93       167
#   talk.politics.mideast       0.99      0.90      0.94       167
#      talk.politics.misc       0.95      0.87      0.91       124
#
#                accuracy                           0.95      1144
#               macro avg       0.95      0.94      0.94      1144
#            weighted avg       0.95      0.95      0.95      1144
#
# ✅ Accuracy: 0.9467
