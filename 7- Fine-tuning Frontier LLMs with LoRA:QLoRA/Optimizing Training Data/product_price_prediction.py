import re
from transformers import AutoTokenizer
from example_builder import TrainingExample



# ─── 1- Task-specific plumbing for price prediction ─────────────────────────────────────────────

# Constants and thresholds
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

QUESTION = "How much does it cost to the nearest dollar?"
PREFIX = "Price is $"

REMOVALS = [
    '"Batteries Included?": "No"', '"Batteries Included?": "Yes"',
    '"Batteries Required?": "No"', '"Batteries Required?": "Yes"',
    "By Manufacturer", "Item", "Date First", "Package", "Number of",
    "Best Sellers", "Number", "Product "
]



# ─── 2- Price cleaning functions ───────────────────────────
def scrubDetails(details: str) -> str:
    """
       Clean up the details string by removing common text that doesn't add value.
    """
    for token in REMOVALS:
        details = details.replace(token, "")
    return details



def scrubText(stuff: str) -> str:
    """
        Clean up the provided text by removing unnecessary characters and whitespace.
        Also remove words that are 7+ chars and contain numbers, as these are likely irrelevant product numbers
    """
    stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
    stuff = stuff.replace(" ,", ",").replace(",,,", ",").replace(",,", ",")
    words = stuff.split(' ')
    select = [word for word in words if len(word) < 7 or not any(char.isdigit() for char in word)]
    return " ".join(select)



# ─── 3- Prepare one shared tokenizer ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)



# ─── 4- Subclass the generic builder ─────────────────────────────────────────
class ProductExample(TrainingExample):


    def __init__(self, raw, price):
        # ─── A) assemble raw combined text ────────────────────────────
        description = "\n".join(raw.get("description", []))
        features = "\n".join(raw.get("features", []))
        details = scrubDetails(raw.get("details", "") or "")
        combined = "\n".join([raw.get("title", ""), description, features, details])

        # ─── B) raw length filter + ceiling ───────────────────────────
        if len(combined) <= MIN_CHARS:
            self.valid = False
            return
        combined = combined[:CEILING_CHARS]

        # ─── C) clean + tokenize truncation ──────────────────────────
        cleaned = f"{scrubText(raw['title'])}\n{scrubText(combined)}"
        tokens = tokenizer.encode(cleaned, add_special_tokens=False)
        if len(tokens) <= MIN_TOKENS:
            self.valid = False
            return
        tokens = tokens[:MAX_TOKENS]
        final_text = tokenizer.decode(tokens)


        # D) invoke the generic initializer
        super().__init__(
            raw = {
                "text": final_text,
                "price": price
            },
            input_key="text",          # picks "text" as the input
            target_key="price",        # picks "price" as the input
            clean_funcs=[],            # already cleaned
            tokenizer=tokenizer,
            template=f"{QUESTION}\n\n{{input}}\n\n{PREFIX}{{target}}.00",
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
            min_chars=MIN_CHARS
        )


