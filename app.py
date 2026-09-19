"""
STEP 3 — Serving API.

The API receives ingredient information and detects allergens.

POST /detect

{
    "ingredients": [
        "wheat flour",
        "sugar",
        "peanut butter",
        "milk powder"
    ]
}

Response:

{
    "allergens": [
        "Milk",
        "Peanuts",
        "Wheat/Gluten"
    ]
}

The system uses:
    1. Explicit ingredient matching
    2. ML-based prediction as supporting evidence

The product name itself is not used for allergen detection.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import re


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


DISPLAY_NAMES = {
    "milk": "Milk",
    "eggs": "Eggs",
    "fish": "Fish",
    "shellfish": "Shellfish",
    "tree_nuts": "Tree Nuts",
    "peanuts": "Peanuts",
    "wheat_gluten": "Wheat/Gluten",
    "soy": "Soy",
    "sesame": "Sesame",
}


# ML threshold.
#
# The previous 0.50 threshold produced false positives,
# such as detecting Soy when the ingredients did not contain soy.
#
# A higher threshold makes the ML layer more conservative.
ML_THRESHOLD = 0.75


# Explicit ingredient keywords.
#
# These directly identify obvious allergen ingredients.
EXPLICIT_ALLERGENS = {

    # ============================================================
    # MILK / DAIRY
    # ============================================================
    "milk": [
        # General
        "milk",
        "dairy",
        "dairy product",
        "dairy products",
        "dairy solids",
        "milk solids",
        "milk ingredients",
        "milk derivative",
        "milk derivatives",

        # Milk powders
        "milk powder",
        "powdered milk",
        "dry milk",
        "dry milk powder",
        "whole milk powder",
        "skim milk powder",
        "skimmed milk powder",
        "nonfat dry milk",
        "nonfat dry milk powder",
        "fat free milk powder",
        "low fat milk powder",

        # Milk concentrates / solids
        "milk concentrate",
        "milk concentrate powder",
        "milk solids nonfat",
        "nonfat milk solids",
        "milk protein",
        "milk proteins",
        "milk protein concentrate",
        "milk protein isolate",

        # Casein
        "casein",
        "caseins",
        "caseinate",
        "caseinates",
        "sodium caseinate",
        "calcium caseinate",
        "potassium caseinate",
        "magnesium caseinate",
        "casein protein",
        "micellar casein",

        # Whey
        "whey",
        "whey powder",
        "whey protein",
        "whey protein concentrate",
        "whey protein isolate",
        "whey solids",
        "whey permeate",
        "sweet whey",
        "acid whey",
        "demineralized whey",
        "whey concentrate",
        "whey isolate",

        # Cream
        "cream",
        "milk cream",
        "cream powder",
        "cream solids",
        "heavy cream",
        "heavy whipping cream",
        "whipping cream",
        "light cream",
        "half and half",
        "half-and-half",
        "sour cream",
        "cultured cream",

        # Butter / milk fat
        "butter",
        "butterfat",
        "butter fat",
        "milk fat",
        "milkfat",
        "anhydrous milk fat",
        "anhydrous butterfat",
        "ghee",

        # Fermented dairy
        "buttermilk",
        "buttermilk powder",
        "yogurt",
        "yoghurt",
        "yogurt powder",
        "yoghurt powder",
        "cultured milk",
        "cultured dairy",

        # Cheese
        "cheese",
        "cheese powder",
        "cheese solids",
        "cheese protein",
        "cheddar",
        "cheddar cheese",
        "mozzarella",
        "mozzarella cheese",
        "parmesan",
        "parmesan cheese",
        "provolone",
        "gouda",
        "swiss cheese",
        "blue cheese",
        "brie",
        "camembert",
        "ricotta",
        "cottage cheese",
        "cream cheese",

        # Processed milk products
        "condensed milk",
        "sweetened condensed milk",
        "evaporated milk",
        "milk beverage",
        "milk concentrate",
        "milk syrup",
    ],


    # ============================================================
    # EGGS
    # ============================================================
    "eggs": [
        # General
        "egg",
        "eggs",
        "whole egg",
        "whole eggs",
        "egg product",
        "egg products",
        "egg solids",
        "egg protein",
        "egg proteins",

        # Yolk
        "egg yolk",
        "egg yolks",
        "yolk",
        "yolks",
        "egg yolk powder",
        "dried egg yolk",
        "dried egg yolks",
        "pasteurized egg yolk",

        # White
        "egg white",
        "egg whites",
        "egg white powder",
        "dried egg white",
        "dried egg whites",
        "pasteurized egg white",

        # Egg powder
        "egg powder",
        "dried egg",
        "dried eggs",
        "whole egg powder",

        # Egg proteins / components
        "albumin",
        "albumen",
        "egg albumin",
        "ovalbumin",
        "ovomucin",
        "ovomucoid",
        "ovotransferrin",
        "ovotransferrin",
        "ovoglobulin",
        "avidin",
        "lysozyme",

        # Egg-containing preparations
        "egg wash",
        "egg wash coating",
        "meringue",
        "meringue powder",
        "mayonnaise",
        "mayo",
        "egg mayonnaise",
        "aioli",
    ],


    # ============================================================
    # FISH
    # ============================================================
    "fish": [
        # General
        "fish",
        "fish protein",
        "fish proteins",
        "fish extract",
        "fish extracts",
        "fish powder",
        "fish meal",
        "fish paste",
        "fish stock",
        "fish broth",
        "fish sauce",
        "fish concentrate",

        # Anchovy
        "anchovy",
        "anchovies",
        "anchovy paste",
        "anchovy extract",
        "anchovy sauce",

        # Tuna
        "tuna",
        "tuna fish",
        "tuna extract",
        "tuna paste",

        # Salmon
        "salmon",
        "salmon fish",
        "salmon extract",
        "salmon powder",

        # Cod
        "cod",
        "codfish",
        "cod fish",
        "cod extract",

        # Mackerel
        "mackerel",
        "mackerel fish",

        # Sardine
        "sardine",
        "sardines",
        "sardine extract",

        # Herring
        "herring",
        "herring fish",

        # Trout
        "trout",
        "trout fish",

        # Tilapia
        "tilapia",

        # Haddock
        "haddock",

        # Pollock
        "pollock",
        "alaska pollock",

        # Halibut
        "halibut",

        # Bass
        "bass fish",

        # Carp
        "carp",

        # Catfish
        "catfish",

        # Mahi
        "mahi mahi",
        "mahi-mahi",

        # Other common fish
        "swordfish",
        "marlin",
        "sardine",
        "eel",
        "pike",
        "perch",
        "snapper",
        "grouper",
        "mullet",
        "hake",
        "sole",
        "flounder",
        "sea bass",
        "whitefish",
    ],


    # ============================================================
    # SHELLFISH
    # ============================================================
    "shellfish": [
        # General
        "shellfish",
        "shellfish protein",
        "shellfish extract",
        "shellfish powder",
        "shellfish paste",
        "shellfish stock",
        "shellfish broth",

        # Shrimp
        "shrimp",
        "shrimps",
        "prawn",
        "prawns",
        "shrimp paste",
        "shrimp powder",
        "shrimp extract",
        "shrimp stock",
        "shrimp broth",
        "shrimp sauce",

        # Crab
        "crab",
        "crabs",
        "crab meat",
        "crab extract",
        "crab paste",
        "crab powder",
        "crab stock",

        # Lobster
        "lobster",
        "lobsters",
        "lobster meat",
        "lobster extract",
        "lobster stock",

        # Crayfish
        "crayfish",
        "crawfish",
        "crawfish meat",
        "crayfish extract",

        # Langoustine
        "langoustine",
        "langoustines",

        # Mussel
        "mussel",
        "mussels",
        "mussel extract",
        "mussel powder",

        # Clam
        "clam",
        "clams",
        "clam extract",
        "clam juice",
        "clam powder",

        # Oyster
        "oyster",
        "oysters",
        "oyster extract",
        "oyster sauce",
        "oyster powder",

        # Scallop
        "scallop",
        "scallops",
        "scallop extract",
        "scallop powder",

        # Other mollusks
        "cockle",
        "cockles",
        "abalone",
        "abalone extract",
        "conch",
        "whelk",
        "periwinkle",
    ],


    # ============================================================
    # TREE NUTS
    # ============================================================
    "tree_nuts": [
        # Almond
        "almond",
        "almonds",
        "almond flour",
        "almond meal",
        "almond powder",
        "almond butter",
        "almond paste",
        "almond milk",
        "almond protein",
        "almond protein powder",
        "almond oil",
        "almond extract",

        # Cashew
        "cashew",
        "cashews",
        "cashew flour",
        "cashew meal",
        "cashew powder",
        "cashew butter",
        "cashew paste",
        "cashew milk",
        "cashew protein",
        "cashew oil",

        # Walnut
        "walnut",
        "walnuts",
        "walnut flour",
        "walnut meal",
        "walnut powder",
        "walnut butter",
        "walnut paste",
        "walnut milk",
        "walnut oil",
        "walnut extract",

        # Hazelnut
        "hazelnut",
        "hazelnuts",
        "hazelnut flour",
        "hazelnut meal",
        "hazelnut powder",
        "hazelnut butter",
        "hazelnut paste",
        "hazelnut milk",
        "hazelnut oil",
        "hazelnut extract",

        # Pistachio
        "pistachio",
        "pistachios",
        "pistachio flour",
        "pistachio meal",
        "pistachio powder",
        "pistachio butter",
        "pistachio paste",
        "pistachio milk",
        "pistachio oil",

        # Pecan
        "pecan",
        "pecans",
        "pecan flour",
        "pecan meal",
        "pecan powder",
        "pecan butter",
        "pecan paste",
        "pecan oil",

        # Macadamia
        "macadamia",
        "macadamia nut",
        "macadamia nuts",
        "macadamia flour",
        "macadamia meal",
        "macadamia butter",
        "macadamia paste",
        "macadamia oil",
        "macadamia milk",

        # Brazil nut
        "brazil nut",
        "brazil nuts",
        "brazilian nut",
        "brazilian nuts",
        "brazil nut oil",

        # Pine nut
        "pine nut",
        "pine nuts",
        "pine kernel",
        "pine kernels",
        "pine nut oil",

        # Other tree nuts
        "chestnut",
        "chestnuts",
        "chestnut flour",
        "chestnut meal",
        "chestnut paste",
        "chestnut oil",

        "coconut",
        "coconut milk",
        "coconut cream",
        "coconut flour",
        "coconut meal",
        "coconut powder",
        "coconut butter",
        "coconut oil",
        "coconut protein",

        "gianduja",
        "praline",
        "nut paste",
        "nut flour",
        "nut meal",
        "nut butter",
        "mixed nuts",
        "mixed tree nuts",
    ],


    # ============================================================
    # PEANUTS
    # ============================================================
    "peanuts": [
        # General
        "peanut",
        "peanuts",
        "peanut product",
        "peanut products",

        # Flour / powder
        "peanut flour",
        "peanut powder",
        "defatted peanut flour",
        "defatted peanut powder",

        # Butter / paste
        "peanut butter",
        "peanut paste",
        "peanut spread",
        "peanut cream",

        # Protein
        "peanut protein",
        "peanut protein isolate",
        "peanut protein concentrate",

        # Oil
        "peanut oil",
        "groundnut oil",
        "arachis oil",

        # Ground / crushed
        "ground peanuts",
        "ground peanut",
        "crushed peanuts",
        "crushed peanut",
        "roasted peanuts",
        "roasted peanut",

        # Alternate terminology
        "groundnut",
        "groundnuts",
        "arachis",
        "arachis hypogaea",
    ],


    # ============================================================
    # WHEAT / GLUTEN
    # ============================================================
    "wheat_gluten": [
        # General
        "wheat",
        "wheat flour",
        "wheat flour powder",
        "wheat product",
        "wheat products",
        "wheat ingredients",
        "wheat derivative",
        "wheat derivatives",

        # Common flour names
        "all-purpose flour",
        "all purpose flour",
        "plain flour",
        "white flour",
        "bread flour",
        "cake flour",
        "pastry flour",
        "self-rising flour",
        "self rising flour",
        "self-raising flour",
        "self raising flour",
        "strong flour",
        "whole wheat flour",
        "whole-wheat flour",
        "wholemeal flour",
        "whole meal flour",
        "graham flour",

        # Wheat components
        "wheat starch",
        "wheat starches",
        "wheat protein",
        "wheat proteins",
        "wheat gluten",
        "vital wheat gluten",
        "wheat germ",
        "wheat germ oil",
        "wheat bran",
        "wheat bran flour",
        "wheat fiber",
        "wheat fibre",

        # Gluten
        "gluten",
        "gluten protein",
        "gluten proteins",
        "vital gluten",

        # Wheat species / grains
        "durum",
        "durum wheat",
        "durum flour",
        "semolina",
        "spelt",
        "spelt flour",
        "einkorn",
        "einkorn flour",
        "emmer",
        "emmer flour",
        "kamut",
        "khorasan wheat",
        "triticale",

        # Wheat-derived foods
        "bread",
        "bread crumbs",
        "breadcrumbs",
        "bread crumb",
        "breading",
        "breaded",
        "cracker",
        "crackers",
        "cookie",
        "cookies",
        "biscuit",
        "biscuits",
        "pasta",
        "macaroni",
        "spaghetti",
        "noodles",
        "wheat noodles",

        # Gluten-containing traditional products
        "seitan",
        "couscous",
        "bulgur",
        "farina",
        "matzo",
        "matzah",
        "malt",
        "wheat malt",
        "wheat beer",

        # Wheat germ / bran
        "wheat germ",
        "wheat bran",
        "wheat bran powder",
        "wheat germ powder",
    ],


    # ============================================================
    # SOY
    # ============================================================
    "soy": [
        # General
        "soy",
        "soya",
        "soybean",
        "soybeans",
        "soy bean",
        "soy beans",
        "soy product",
        "soy products",
        "soy derivative",
        "soy derivatives",

        # Flour / powder
        "soy flour",
        "soya flour",
        "soy powder",
        "soybean flour",
        "soybean powder",

        # Protein
        "soy protein",
        "soy protein isolate",
        "soy protein concentrate",
        "soy protein hydrolysate",
        "hydrolyzed soy protein",
        "hydrolysed soy protein",
        "soy peptide",
        "soy peptides",

        # Soy milk
        "soy milk",
        "soya milk",
        "soy beverage",
        "soy drink",

        # Soy oil
        "soybean oil",
        "soy bean oil",
        "soya oil",
        "soy oil",

        # Lecithin
        "soy lecithin",
        "soya lecithin",
        "soybean lecithin",

        # Sauces / fermented products
        "soy sauce",
        "soya sauce",
        "shoyu",
        "tamari",
        "miso",
        "miso paste",
        "miso powder",
        "fermented soybean",
        "fermented soybeans",

        # Soy foods
        "tofu",
        "bean curd",
        "soy curd",
        "tempeh",
        "edamame",
        "edamame beans",
        "soy nuts",
        "soy nut",
        "roasted soybeans",
        "textured soy protein",
        "textured vegetable protein",
        "TVP",

        # Other
        "soy meal",
        "soybean meal",
        "soy concentrate",
        "soy extract",
        "soybean extract",
    ],


    # ============================================================
    # SESAME
    # ============================================================
    "sesame": [
        # General
        "sesame",
        "sesame seed",
        "sesame seeds",
        "sesame product",
        "sesame products",

        # Flour / meal / powder
        "sesame flour",
        "sesame meal",
        "sesame powder",
        "ground sesame",
        "ground sesame seeds",
        "crushed sesame",

        # Oil
        "sesame oil",
        "sesame seed oil",
        "sesame oil extract",

        # Paste
        "sesame paste",
        "tahini",
        "tahina",
        "sesame butter",

        # Protein
        "sesame protein",
        "sesame protein isolate",
        "sesame protein concentrate",

        # Other
        "sesame extract",
        "sesame seed extract",
        "sesame hull",
        "sesame bran",
        "sesame meal powder",
    ],
}


vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")


app = FastAPI()


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
#
# Allows the Flutter Web application to call the API from
# the browser.
#
# Postman does not enforce browser CORS rules, which is why
# the API could work in Postman while Flutter Web returned:
#
# "ClientException: Failed to fetch"
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    ingredients: List[str]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_explicit_allergens(ingredients: List[str]):
    detected = set()

    for ingredient in ingredients:
        text = normalize_text(ingredient)

        for allergen, keywords in EXPLICIT_ALLERGENS.items():
            for keyword in keywords:
                if keyword in text:
                    detected.add(allergen)
                    break

    return detected


@app.post("/detect")
def detect(req: DetectRequest):
    # Remove empty ingredients.
    ingredients = [
        ingredient.strip()
        for ingredient in req.ingredients
        if ingredient.strip()
    ]

    if not ingredients:
        return {
            "allergens": []
        }

    # Combine all ingredients for the ML model.
    text = ". ".join(ingredients)

    X = vectorizer.transform([text])

    probabilities = model.predict_proba(X)[0]

    # ---------------------------------------------------------
    # STEP 1 — Explicit ingredient detection
    # ---------------------------------------------------------
    #
    # If an ingredient directly contains an allergen keyword,
    # report it immediately.
    #
    # Example:
    # "peanut butter" -> Peanuts
    # "milk powder"  -> Milk
    # "wheat flour"  -> Wheat/Gluten
    #
    detected = detect_explicit_allergens(ingredients)

    # ---------------------------------------------------------
    # STEP 2 — ML supporting detection
    # ---------------------------------------------------------
    #
    # Only add an ML prediction when the probability reaches
    # the higher 0.75 threshold.
    #
    # This prevents weaker predictions from creating too many
    # false positives.
    #
    for allergen, probability in zip(
        ALLERGEN_COLUMNS,
        probabilities,
    ):
        if probability >= ML_THRESHOLD:
            detected.add(allergen)

    # ---------------------------------------------------------
    # STEP 3 — Consistent output order
    # ---------------------------------------------------------

    allergens = [
        DISPLAY_NAMES[allergen]
        for allergen in ALLERGEN_COLUMNS
        if allergen in detected
    ]

    return {
        "allergens": allergens
    }


@app.get("/")
def health():
    return {
        "status": "ok"
    }