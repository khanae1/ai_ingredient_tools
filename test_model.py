import joblib

model = joblib.load("model_gemini.joblib")
vectorizer = joblib.load("vectorizer_gemini.joblib")

ALLERGENS = [
    "Milk",
    "Eggs",
    "Fish",
    "Shellfish",
    "Tree Nuts",
    "Peanuts",
    "Wheat/Gluten",
    "Soy",
    "Sesame",
]

TEST_CASES = [
    "douchi",
    "miso",
    "tempeh",
    "tahini",
    "whey",
    "anchovy paste",
    "soy sauce",
    "peanut oil",
    "wheat flour",
    "almond",
    "apple",
    "banana",
]

X = vectorizer.transform(TEST_CASES)
probabilities = model.predict_proba(X)

for ingredient, probs in zip(TEST_CASES, probabilities):
    found = [
        f"{ALLERGENS[i]} ({probs[i]:.2f})"
        for i in range(len(ALLERGENS))
        if probs[i] >= 0.50
    ]

    print(f"\n{ingredient}")
    print("  " + (", ".join(found) if found else "No allergen detected"))