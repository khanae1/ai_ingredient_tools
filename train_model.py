import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE = Path(__file__).parent

DATASET = BASE / "gemini_training_data.csv"
MODEL_OUT = BASE / "model_gemini.joblib"
VECTORIZER_OUT = BASE / "vectorizer_gemini.joblib"

ALLERGEN_COLUMNS = [
    "milk",
    "eggs",
    "fish",
    "shellfish",
    "tree_nuts",
    "peanuts",
    "wheat_gluten",
    "soy",
    "sesame",
]


print("Loading dataset...")
df = pd.read_csv(DATASET)

df["ingredient"] = (
    df["ingredient"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[df["ingredient"] != ""].copy()
df = df.drop_duplicates(subset=["ingredient"])

print(f"Rows: {len(df)}")
print(f"Allergen labels: {len(ALLERGEN_COLUMNS)}")

X_text = df["ingredient"]

y = df[ALLERGEN_COLUMNS].astype(int)

print("\nLabel counts:")
print(y.sum().sort_values(ascending=False))

print("\nTraining vectorizer...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True,
)

X = vectorizer.fit_transform(X_text)

print(f"Feature count: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining OneVsRest LogisticRegression model...")

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
)

model.fit(X_train, y_train)

print("\nModel trained.")

print("\nEvaluating candidate model...")

y_pred = model.predict(X_test)

for index, allergen in enumerate(ALLERGEN_COLUMNS):
    print(f"\n===== {allergen} =====")

    print(
        classification_report(
            y_test.iloc[:, index],
            y_pred[:, index],
            zero_division=0,
        )
    )

print("\nSaving candidate model...")

joblib.dump(model, MODEL_OUT)
joblib.dump(vectorizer, VECTORIZER_OUT)

print("\n==========================================")
print(" Candidate model created!")
print("==========================================")
print(f"Model:      {MODEL_OUT}")
print(f"Vectorizer: {VECTORIZER_OUT}")