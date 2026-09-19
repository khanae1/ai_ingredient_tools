---
title: Allergen Detector
emoji: 🥜
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
---

# Allergen Detector

TF-IDF + Logistic Regression multi-label classifier trained on Open Food
Facts data. POST /detect with {"name": "...", "notes": "..."} to get back
{"allergens": [...]}.

Model files (vectorizer.joblib, model.joblib) must be uploaded into this
Space alongside this code — see train_model.py in the training project.
