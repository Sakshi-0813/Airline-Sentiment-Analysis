
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from preprocessing import preprocess_series

df = pd.read_csv("C:/Users/ASUS ROG STRIX/OneDrive/Desktop/twitterdataset/Tweets.csv")
df = df.dropna(subset=["text", "airline_sentiment"]).copy()

# Remove duplicate texts with conflicting labels first
conflicts = df.groupby("text")["airline_sentiment"].nunique()
conflicting_texts = conflicts[conflicts > 1].index
df = df[~df["text"].isin(conflicting_texts)].copy()

df = df.drop_duplicates(subset=["text"])

X = df["text"].astype(str)
y = df["airline_sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(
        class_weight="balanced"
    )
}

results = []
trained_models = {}

for name, classifier in classifiers.items():
    pipeline = Pipeline([
        ("preprocessing", FunctionTransformer(
            preprocess_series,
            validate=False
        )),
        ("tfidf", TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro F1": f1_score(
            y_test, predictions, average="macro"
        )
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print(classification_report(y_test, predictions))

results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df.sort_values("Macro F1", ascending=False))

best_name = results_df.sort_values(
    "Macro F1", ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_name]
joblib.dump(best_model, "sentiment_model.pkl")

print("\nSelected model:", best_name)
print("Model saved successfully.")
