import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

from allergens import detect_allergens


BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# GEMINI
# Used ONLY for ingredient identification.
# /detect does NOT use Gemini.
# ============================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODELS = [
    "gemini-3.1-flash-lite",
]


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Ingredient & Allergen API"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class IdentifyRequest(BaseModel):
    name: str


class IdentifyResult(BaseModel):
    ingredients: list[str]


class DetectRequest(BaseModel):
    ingredients: list[str]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "Ingredient & Allergen API is running",
    }


# ============================================================
# INGREDIENT IDENTIFICATION
# ============================================================
# Gemini is allowed here.
#
# Example:
#
# POST /identify
# {
#     "name": "Burger"
# }
#
# ============================================================

@app.post("/identify")
def identify(req: IdentifyRequest):

    name = req.name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name is required",
        )

    prompt = (
        f'List the ingredients of this food product or ingredient: "{name}". '
        "If it is a packaged product, list its typical ingredients. "
        "If it is a single ingredient, list what it is commonly made of or derived from. "
        "Use short lowercase names, no quantities. "
        "If you don't recognize it, return an empty list."
    )

    last_error = None

    for model_name in MODELS:

        for attempt in range(2):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=IdentifyResult,
                        temperature=0,
                    ),
                )

                return {
                    "name": name,
                    "ingredients": response.parsed.ingredients,
                }

            except Exception as e:

                last_error = e

                if attempt == 0:
                    time.sleep(1)

    raise HTTPException(
        status_code=502,
        detail=f"Gemini error: {last_error}",
    )


# ============================================================
# ALLERGEN DETECTION
# ============================================================
#
# IMPORTANT:
#
# THIS ENDPOINT DOES NOT CALL GEMINI.
#
# It uses:
#
#     allergens.py
#          ↓
#     local TF-IDF vectorizer
#          ↓
#     local OneVsRest model
#          ↓
#     local explicit allergen rules
#
# Therefore this endpoint works without:
#
#     - Gemini API
#     - internet access
#     - Gemini quota
#
# ============================================================

@app.post("/detect")
def detect(req: DetectRequest):

    ingredients = [
        ingredient.strip()
        for ingredient in req.ingredients
        if isinstance(ingredient, str)
        and ingredient.strip()
    ]

    if not ingredients:
        return {
            "ingredients": [],
            "allergens": [],
        }

    allergens = detect_allergens(ingredients)

    return {
        "ingredients": ingredients,
        "allergens": allergens,
    }