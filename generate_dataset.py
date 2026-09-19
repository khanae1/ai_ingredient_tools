import csv
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE = Path(__file__).parent

load_dotenv(BASE / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.1-flash-lite"

OUTPUT_FILE = BASE / "gemini_training_data.csv"

BATCH_SIZE = 25

# Keep this conservative because of the 15 RPM limit.
BATCH_DELAY = 5

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


# ---------------------------------------------------------
# Extract existing vocabulary from allergens.py
# ---------------------------------------------------------

def extract_keywords():
    allergens_file = BASE / "allergens.py"

    if not allergens_file.exists():
        raise FileNotFoundError(
            "allergens.py was not found."
        )

    text = allergens_file.read_text(
        encoding="utf-8"
    )

    results = []

    current_allergen = None

    for line in text.splitlines():

        stripped = line.strip()

        for allergen in ALLERGEN_COLUMNS:

            if stripped.startswith(
                f'"{allergen}": ['
            ):
                current_allergen = allergen
                break

        if current_allergen is None:
            continue

        if stripped == "],":
            current_allergen = None
            continue

        match = re.match(
            r'"([^"]+)",?',
            stripped
        )

        if match:

            ingredient = match.group(1).strip()

            if ingredient:

                results.append(
                    (ingredient, current_allergen)
                )

    return results


# ---------------------------------------------------------
# Neutral ingredients
#
# These are important because the model must learn what
# does NOT indicate an allergen.
# ---------------------------------------------------------

NEUTRAL_INGREDIENTS = [

    # Basic ingredients
    "water",
    "salt",
    "sugar",
    "potato",
    "carrot",
    "onion",
    "tomato",
    "garlic",
    "ginger",

    # Herbs
    "basil",
    "parsley",
    "coriander",
    "cilantro",
    "black pepper",
    "white pepper",
    "turmeric",
    "paprika",
    "chili",
    "cinnamon",
    "nutmeg",
    "clove",
    "cumin",
    "oregano",
    "rosemary",
    "thyme",
    "dill",
    "bay leaf",
    "sage",
    "tarragon",

    # Fruits
    "apple",
    "banana",
    "mango",
    "pineapple",
    "orange",
    "lemon",
    "lime",
    "strawberry",
    "blueberry",
    "raspberry",
    "watermelon",
    "papaya",
    "avocado",
    "grape",
    "peach",
    "pear",
    "plum",

    # Vegetables
    "broccoli",
    "cauliflower",
    "cabbage",
    "spinach",
    "lettuce",
    "cucumber",
    "zucchini",
    "eggplant",
    "celery",
    "radish",
    "beetroot",
    "sweet potato",
    "pumpkin",
    "squash",
    "mushroom",

    # Starches / grains
    "potato starch",
    "corn starch",
    "tapioca starch",
    "cassava",
    "rice",
    "rice flour",
    "corn",
    "corn flour",
    "corn syrup",

    # Oils
    "olive oil",
    "sunflower oil",
    "canola oil",
    "vegetable oil",
    "avocado oil",

    # Other
    "cocoa powder",
    "cocoa butter",
    "coffee",
    "tea",
    "baking soda",
    "baking powder",
    "citric acid",
    "maltodextrin",
    "caramel color",
    "pectin",
    "agar",
    "gelatin",
    "xanthan gum",
    "guar gum",
]


# ---------------------------------------------------------
# SEMANTIC INGREDIENT VOCABULARY
#
# These are intentionally difficult cases.
#
# The purpose is NOT to hardcode their answers.
# Gemini will determine their labels.
#
# This gives the ML model vocabulary that it previously
# never saw or barely saw.
# ---------------------------------------------------------

SEMANTIC_INGREDIENTS = [

    # -----------------------------------------------------
    # SOY
    # -----------------------------------------------------
    "douchi",
    "douchi paste",
    "douchi beans",
    "fermented douchi",
    "Chinese douchi",
    "Chinese douchi beans",
    "Chinese fermented douchi",
    "fermented black douchi",
    "fermented black bean douchi",
    "salted douchi",
    "salted fermented black beans",
    "fermented black soybean",
    "fermented black soybeans",
    "fermented black soybean paste",
    "Chinese fermented black soybean",
    "Chinese fermented black soybeans",
    "douchi",
    "fermented black beans",
    "fermented black bean paste",
    "Chinese fermented black beans",
    "Chinese fermented soybean",
    "fermented soybeans",
    "fermented soybean paste",
    "miso",
    "miso paste",
    "miso powder",
    "white miso",
    "red miso",
    "yellow miso",
    "awase miso",
    "hatcho miso",
    "tempeh",
    "tempe",
    "natto",
    "soybean curd",
    "tofu",
    "silken tofu",
    "firm tofu",
    "bean curd",
    "soybean paste",
    "soybean powder",
    "soy protein",
    "soy protein isolate",
    "soy protein concentrate",
    "textured soy protein",
    "textured vegetable protein",
    "TVP",
    "hydrolyzed soy protein",
    "soy lecithin",
    "soy flour",
    "soy milk",
    "soy beverage",
    "soy yogurt",
    "edamame",
    "soybeans",
    "soybean oil",
    "soy sauce",
    "shoyu",
    "tamari",
    "black bean sauce",
    "hoisin sauce",
    "chickpea miso",

    # -----------------------------------------------------
    # MILK / DAIRY
    # -----------------------------------------------------

    "whey",
    "whey powder",
    "whey protein",
    "whey protein concentrate",
    "whey protein isolate",
    "acid whey",
    "sweet whey",
    "casein",
    "casein protein",
    "sodium caseinate",
    "calcium caseinate",
    "potassium caseinate",
    "milk powder",
    "whole milk powder",
    "skim milk powder",
    "nonfat dry milk",
    "dry milk solids",
    "milk solids",
    "milk protein",
    "milk protein concentrate",
    "milk protein isolate",
    "lactose",
    "lactalbumin",
    "lactoglobulin",
    "butter",
    "butter oil",
    "ghee",
    "cream",
    "heavy cream",
    "whipping cream",
    "sour cream",
    "cream cheese",
    "cheese",
    "cheddar cheese",
    "parmesan cheese",
    "mozzarella",
    "milk chocolate",

    # -----------------------------------------------------
    # EGGS
    # -----------------------------------------------------

    "egg",
    "whole egg",
    "egg white",
    "egg yolk",
    "egg powder",
    "dried egg",
    "dried egg white",
    "dried egg yolk",
    "albumin",
    "ovalbumin",
    "ovomucin",
    "ovomucoid",
    "egg protein",
    "egg lecithin",
    "mayonnaise",
    "aioli",
    "meringue",
    "custard",

    # -----------------------------------------------------
    # FISH
    # -----------------------------------------------------

    "anchovy",
    "anchovy paste",
    "anchovy extract",
    "fish sauce",
    "fish stock",
    "fish broth",
    "fish extract",
    "fish protein",
    "fish gelatin",
    "tuna",
    "tuna paste",
    "salmon",
    "salmon extract",
    "sardine",
    "sardine paste",
    "mackerel",
    "cod",
    "cod liver",
    "herring",
    "herring paste",
    "trout",
    "tilapia",
    "pollock",
    "haddock",
    "sardine sauce",
    "bonito",
    "bonito flakes",
    "dried bonito",
    "katsuobushi",
    "surimi",

    # -----------------------------------------------------
    # SHELLFISH
    # -----------------------------------------------------

    "shrimp",
    "prawn",
    "crab",
    "crab meat",
    "crab extract",
    "lobster",
    "lobster extract",
    "crayfish",
    "crawfish",
    "oyster",
    "oyster extract",
    "oyster sauce",
    "mussel",
    "mussel extract",
    "clam",
    "clam extract",
    "scallop",
    "scallop extract",
    "shellfish extract",
    "shellfish stock",
    "shrimp paste",
    "shrimp powder",
    "dried shrimp",
    "shrimp stock",
    "prawn paste",
    "crab paste",

    # -----------------------------------------------------
    # TREE NUTS
    # -----------------------------------------------------

    "almond",
    "almond flour",
    "almond meal",
    "almond butter",
    "almond paste",
    "almond milk",
    "almond extract",
    "cashew",
    "cashew butter",
    "cashew paste",
    "cashew milk",
    "walnut",
    "walnut oil",
    "walnut flour",
    "pecan",
    "pecan butter",
    "hazelnut",
    "hazelnut paste",
    "hazelnut butter",
    "hazelnut extract",
    "pistachio",
    "pistachio paste",
    "macadamia",
    "macadamia nut oil",
    "brazil nut",
    "brazil nut oil",
    "pine nut",
    "pine nuts",
    "chestnut",
    "chestnut flour",
    "praline",

    # -----------------------------------------------------
    # PEANUTS
    # -----------------------------------------------------

    "peanut",
    "peanut flour",
    "peanut meal",
    "peanut butter",
    "peanut paste",
    "peanut powder",
    "peanut protein",
    "peanut oil",
    "ground peanuts",
    "roasted peanuts",
    "peanut sauce",
    "peanut satay sauce",
    "satay sauce",

    # -----------------------------------------------------
    # WHEAT / GLUTEN
    # -----------------------------------------------------

    "wheat",
    "wheat flour",
    "whole wheat flour",
    "white wheat flour",
    "bread flour",
    "all purpose flour",
    "plain flour",
    "strong flour",
    "cake flour",
    "pastry flour",
    "durum wheat",
    "semolina",
    "farina",
    "bulgur",
    "couscous",
    "wheat bran",
    "wheat germ",
    "wheat starch",
    "wheat protein",
    "wheat gluten",
    "vital wheat gluten",
    "seitan",
    "hydrolyzed wheat protein",
    "wheat malt",
    "malt",
    "barley",
    "barley flour",
    "barley malt",
    "barley malt extract",
    "rye",
    "rye flour",
    "rye bread",
    "spelt",
    "spelt flour",
    "kamut",
    "farro",
    "triticale",
    "brewer's yeast",
    "beer",
    "soy sauce",
    "shoyu",
    "bread crumbs",
    "breadcrumbs",
    "panko",
    "cracker crumbs",
    "pasta",
    "noodles",
    "wheat noodles",
    "egg noodles",
    "udon",
    "ramen noodles",

    # -----------------------------------------------------
    # SESAME
    # -----------------------------------------------------

    "sesame",
    "sesame seed",
    "sesame seeds",
    "sesame oil",
    "sesame paste",
    "sesame flour",
    "tahini",
    "tahini paste",
    "sesame butter",
    "sesame protein",
    "halva",
    "hummus with tahini",
    "sesame dressing",
    "sesame sauce",


        # -----------------------------------------------------
    # Additional semantic soy variations
    # -----------------------------------------------------

    "douchi paste",
    "fermented black soybean",
    "fermented black soybeans",
    "fermented black soybean paste",
    "Chinese black bean paste",
    "Chinese fermented bean paste",
    "fermented bean paste",
    "fermented soybean product",
    "fermented soybean food",
    "fermented soybean ingredient",
    "fermented soy bean paste",
    "fermented soybean cake",
    "soybean fermentation",
    "fermented bean curd",
    "fermented tofu",
    "fermented tofu paste",

    "tempeh cake",
    "soy tempeh",
    "fermented soy cake",
    "fermented soybean cake",
    "traditional tempeh",
    "fermented soybean blocks",
    "fermented soybean cakes",

    # -----------------------------------------------------
    # Additional semantic sesame variations
    # -----------------------------------------------------

    "sesame seed paste",
    "ground sesame",
    "ground sesame seeds",
    "ground sesame paste",
    "ground sesame butter",
    "sesame seed butter",
    "sesame butter paste",
    "sesame puree",
    "sesame seed puree",
    "sesame seed meal",
    "sesame meal",
    "sesame seed powder",
    "ground sesame powder",
    "sesame paste spread",

    # -----------------------------------------------------
    # Additional milk / dairy variations
    # -----------------------------------------------------

    "milk-derived whey",
    "whey solids",
    "whey concentrate",
    "whey isolate",
    "milk-derived casein",
    "casein concentrate",
    "casein isolate",
    "caseinate",
    "milk caseinate",
    "dairy protein",
    "dairy protein concentrate",
    "dairy protein isolate",

    # -----------------------------------------------------
    # Additional fish variations
    # -----------------------------------------------------

    "anchovy extract",
    "anchovy concentrate",
    "anchovy seasoning",
    "fermented anchovy",
    "dried anchovy",
    "anchovy powder",
    "fish seasoning",
    "fish concentrate",
    "fish protein concentrate",

    # -----------------------------------------------------
    # Additional shellfish variations
    # -----------------------------------------------------

    "shrimp extract",
    "shrimp concentrate",
    "shrimp seasoning",
    "dried shrimp powder",
    "crab extract",
    "crab concentrate",
    "oyster concentrate",
    "oyster seasoning",
    "clam extract",
    "clam concentrate",

    # -----------------------------------------------------
    # Additional peanut variations
    # -----------------------------------------------------

    "ground peanut",
    "ground peanuts",
    "roasted peanut",
    "roasted peanuts",
    "peanut meal",
    "peanut flour",
    "peanut protein powder",
    "peanut extract",

    # -----------------------------------------------------
    # Additional tree-nut variations
    # -----------------------------------------------------

    "ground almond",
    "ground almonds",
    "almond meal",
    "almond powder",
    "ground cashew",
    "ground cashews",
    "cashew powder",
    "ground walnut",
    "ground walnuts",
    "walnut powder",
    "hazelnut powder",
    "pistachio powder",
    "pecan powder",

    # -----------------------------------------------------
    # Additional wheat / gluten variations
    # -----------------------------------------------------

    "wheat protein isolate",
    "wheat protein concentrate",
    "wheat flour blend",
    "wheat-based flour",
    "gluten flour",
    "gluten protein",
    "gluten concentrate",
    "wheat dough",
    "wheat bread",
    "wheat crumbs",
    "wheat cereal",

    # -----------------------------------------------------
    # Additional egg variations
    # -----------------------------------------------------

    "egg protein powder",
    "egg white powder",
    "egg yolk powder",
    "dried whole egg",
    "dried egg protein",
    "egg albumen",
    "egg albumin",

]


# ---------------------------------------------------------
# Gemini structured response
# ---------------------------------------------------------

class IngredientLabel(BaseModel):
    ingredient: str
    allergens: list[str]


class BatchResult(BaseModel):
    items: list[IngredientLabel]


# ---------------------------------------------------------
# Ask Gemini to label a batch
# ---------------------------------------------------------

def classify_batch(ingredients):

    prompt = f"""
You are creating a high-quality training dataset for a
food allergen detection machine-learning model.

For every ingredient below, identify which allergen categories
are actually supported by the ingredient itself.

Allowed allergen categories ONLY:

- milk
- eggs
- fish
- shellfish
- tree_nuts
- peanuts
- wheat_gluten
- soy
- sesame

Important:

This is semantic classification.

DO NOT use simple substring matching.

Understand the actual food ingredient, traditional name,
synonym, derivative, or source.

Examples:

douchi -> soy
fermented black beans -> soy
miso -> soy
tempeh -> soy
tahini -> sesame
whey -> milk
casein -> milk
anchovy paste -> fish
oyster -> shellfish
peanut oil -> peanuts
almond flour -> tree_nuts
wheat flour -> wheat_gluten

Do NOT infer an allergen merely because a word happens to
contain another word.

For example, unrelated words must not receive an allergen
just because their spelling resembles an allergen name.

Rules:

1. Analyze the actual meaning and source of the ingredient.
2. Recognize traditional food names and synonyms.
3. Recognize ingredient derivatives.
4. Recognize common food-industry ingredient terminology.
5. Only return allergens reasonably supported by the ingredient.
6. If there is no supported allergen, return [].
7. Return exactly one result for every ingredient.
8. Preserve the ingredient text exactly.
9. Return ONLY the allowed category names.
10. Do not invent new allergen categories.

Ingredients to classify:

{ingredients}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchResult,
            temperature=0,
        ),
    )

    return response.parsed.items


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print()
    print("==========================================")
    print(" Gemini Training Dataset Generator")
    print("==========================================")
    print()

    # -----------------------------------------------------
    # Existing vocabulary
    # -----------------------------------------------------

    extracted = extract_keywords()

    print(
        f"Extracted {len(extracted)} existing "
        f"allergen keywords from allergens.py."
    )

    ingredients = {}

    # Existing allergen vocabulary.
    #
    # IMPORTANT:
    # The labels extracted here are NOT automatically trusted
    # as the final answer.
    #
    # Gemini will classify everything again.
    for ingredient, _ in extracted:

        ingredients.setdefault(
            ingredient.lower(),
            set()
        )

    # -----------------------------------------------------
    # Add neutral vocabulary
    # -----------------------------------------------------

    for ingredient in NEUTRAL_INGREDIENTS:

        ingredients.setdefault(
            ingredient.lower(),
            set()
        )

    # -----------------------------------------------------
    # Add semantic vocabulary
    # -----------------------------------------------------

    for ingredient in SEMANTIC_INGREDIENTS:

        ingredients.setdefault(
            ingredient.lower(),
            set()
        )

    ingredient_list = sorted(
        ingredients.keys()
    )

    print(
        f"Total unique ingredients: "
        f"{len(ingredient_list)}"
    )

    total_batches = (
        len(ingredient_list)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    print()
    print(
        f"Gemini will process approximately "
        f"{total_batches} batches."
    )

    print()
    print(
        "Gemini is being used ONLY to create "
        "training labels."
    )

    print(
        "The resulting model will run locally "
        "without Gemini."
    )

    print()

    rows = []

    # -----------------------------------------------------
    # Gemini classification
    # -----------------------------------------------------

    for start in range(
        0,
        len(ingredient_list),
        BATCH_SIZE
    ):

        batch = ingredient_list[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        print(
            f"[{batch_number}/{total_batches}] "
            f"Sending {len(batch)} ingredients..."
        )

        try:

            results = classify_batch(
                batch
            )

            result_map = {
                item.ingredient.lower(): item.allergens
                for item in results
            }

            for ingredient in batch:

                allergens = result_map.get(
                    ingredient.lower(),
                    []
                )

                # Only accept labels that belong to our
                # controlled vocabulary.
                allergens = [
                    allergen
                    for allergen in allergens
                    if allergen in ALLERGEN_COLUMNS
                ]

                row = {
                    "ingredient": ingredient,
                }

                for allergen in ALLERGEN_COLUMNS:

                    row[allergen] = (
                        1
                        if allergen in allergens
                        else 0
                    )

                rows.append(row)

            print(
                "    ✓ Batch completed."
            )

        except Exception as e:

            print(
                f"    ✗ Batch failed: {e}"
            )

        # -------------------------------------------------
        # Respect Gemini RPM limit
        # -------------------------------------------------

        if (
            start + BATCH_SIZE
            < len(ingredient_list)
        ):

            print(
                f"    Waiting {BATCH_DELAY}s..."
            )

            time.sleep(
                BATCH_DELAY
            )

    # -----------------------------------------------------
    # Remove duplicate rows
    # -----------------------------------------------------

    unique_rows = {}

    for row in rows:

        ingredient = (
            row["ingredient"]
            .strip()
            .lower()
        )

        unique_rows[ingredient] = row

    rows = list(
        unique_rows.values()
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    fieldnames = [
        "ingredient",
        *ALLERGEN_COLUMNS,
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("==========================================")
    print(" Dataset generation complete!")
    print("==========================================")
    print()

    print(
        f"Rows written: {len(rows)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()

    # Show label counts.
    print("Label counts:")

    for allergen in ALLERGEN_COLUMNS:

        count = sum(
            row[allergen] == 1
            for row in rows
        )

        print(
            f"  {allergen}: {count}"
        )

    print()


if __name__ == "__main__":
    main()