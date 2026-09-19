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


# ============================================================
# Environment / Gemini
# ============================================================

load_dotenv(Path(__file__).parent / ".env")

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=gemini_api_key)

# Tried in order.
MODELS = [
    "gemini-3.1-flash-lite",
]


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="AI Ingredient & Allergen Detection API"
)


# Open CORS for now; tighten it before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response Models
# ============================================================

class IdentifyRequest(BaseModel):
    name: str


class IdentifyResult(BaseModel):
    ingredients: list[str]


class DetectRequest(BaseModel):
    ingredients: list[str]


class DetectResult(BaseModel):
    allergens: list[str]


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "Ingredient & Allergen API is running",
    }


# ============================================================
# AI INGREDIENT IDENTIFIER
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
        "If it is a single ingredient, list what it is commonly made of "
        "or derived from. "
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
                time.sleep(1)

    raise HTTPException(
        status_code=502,
        detail=f"Gemini error: {last_error}",
    )


# ============================================================
# GEMINI PRIMARY ALLERGEN DETECTOR
# ============================================================

def detect_allergens_with_gemini(
    ingredients: list[str],
) -> list[str]:

    cleaned = [
        ingredient.strip()
        for ingredient in ingredients
        if ingredient and ingredient.strip()
    ]

    if not cleaned:
        return []

    prompt = f"""
You are the primary allergen analyzer for a food information system.

Analyze the following ingredient list and identify allergens from ONLY
these categories:

- Milk
- Eggs
- Fish
- Shellfish
- Tree Nuts
- Peanuts
- Wheat/Gluten
- Soy
- Sesame

Ingredients:
{cleaned}

Important rules:

1. Determine allergens based on the actual meaning and source of the
   ingredient, not simple keyword matching.

2. Do NOT identify an allergen merely because an allergen word happens
   to appear inside another word.

3. Consider ingredient context and common food ingredient terminology.

4. Do not invent ingredients that are not present.

5. Do not assume that an ingredient is an allergen solely because its
   name resembles an allergen.

6. Only return allergens that are reasonably supported by the provided
   ingredients.

7. If no supported allergens are present, return an empty list.

8. Return ONLY the allergen category names from the allowed list.
"""

    last_error = None

    for model_name in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DetectResult,
                        temperature=0,
                    ),
                )

                return response.parsed.allergens

            except Exception as e:
                last_error = e
                time.sleep(1)

    raise HTTPException(
        status_code=502,
        detail=f"Gemini allergen detection error: {last_error}",
    )


# ============================================================
# ALLERGEN DETECTION
#
# Gemini = PRIMARY / TRUSTED RESULT
# Trained ML model = COMPARISON ONLY
# ============================================================

@app.post("/detect")
def detect(req: DetectRequest):

    ingredients = [
        ingredient.strip()
        for ingredient in req.ingredients
        if ingredient and ingredient.strip()
    ]

    if not ingredients:
        return {
            "ingredients": [],
            "gemini_allergens": [],
            "model_allergens": [],
            "allergens": [],
            "agreement": True,
        }

    # --------------------------------------------------------
    # PRIMARY:
    # Gemini performs the actual contextual allergen analysis.
    # --------------------------------------------------------

    gemini_allergens = detect_allergens_with_gemini(
        ingredients
    )

    # --------------------------------------------------------
    # COMPARISON:
    # Existing Open Food Facts-trained model + explicit rules.
    #
    # This result is NOT used as the final answer.
    # --------------------------------------------------------

    model_allergens = detect_allergens(
        ingredients
    )

    # --------------------------------------------------------
    # Compare the two results.
    # Gemini remains the final/trusted result.
    # --------------------------------------------------------

    agreement = (
        set(gemini_allergens)
        == set(model_allergens)
    )

    return {
        "ingredients": ingredients,

        # Gemini's primary determination.
        "gemini_allergens": gemini_allergens,

        # Existing trained model / rule-based result.
        "model_allergens": model_allergens,

        # FINAL RESULT:
        # Gemini is trusted for the final answer.
        "allergens": gemini_allergens,

        # Useful for testing / presentation.
        "agreement": agreement,
    }