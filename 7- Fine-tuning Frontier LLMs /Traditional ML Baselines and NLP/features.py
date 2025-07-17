import json
from collections import Counter




############## Attach raw JSON → item.features ##############
def attach_raw_features(items):
    """Parse the JSON in .details into a dict on each item."""
    for item in items:
        item.features = json.loads(item.details)



def most_common_feature_keys(items, n=40):
    feature_counts = Counter(key for item in items for key in item.features.keys())
    return feature_counts.most_common(n)



############## Weight extraction + default ##############
def get_weight(item):
    """Extract Item Weight in pounds (or None)."""
    weight_str = item.features.get("Item Weight")
    if weight_str:
        parts = weight_str.split(" ")
        amount = float(parts[0])
        unit = parts[1].lower()
        if unit == "pounds":
            return amount
        elif unit == "ounces":
            return amount / 16
        elif unit == "grams":
            return amount / 453.592
        elif unit == "milligrams":
            return amount / 453592
        elif unit == "kilograms":
            return amount / 0.453592
        elif unit == "hundredths" and parts[2].lower()=="pounds":
            return amount / 100
        else:
            print(weight_str)
    return None



def compute_average_weight(items):
    """Compute mean weight over all items with a valid weight."""
    # Build list of all weights
    weights = [w for w in (get_weight(i) for i in items) if w is not None]
    return sum(weights) / len(weights) if weights else 0.0



def get_weight_or_default(item, default):
    """Pick the item’s own weight when available, otherwise average weight."""
    weight = get_weight(item)   # weight for each item
    return weight if weight is not None else default




############## Bestseller rank + default ##############
def get_rank(item):
    """Take the average of *per-item* in Best Sellers Rank dict.
       If the item has multiple “Best Sellers” categories,
       take the mean of those category ranks.
    """
    rank_dict = item.features.get("Best Sellers Rank")
    if rank_dict:
        ranks = rank_dict.values()
        return sum(ranks) / len(ranks)
    return None


def compute_average_rank(items):
    """ what’s the mean of those per-item averages across all items? """
    # first build a list of all the per-item ranks
    ranks = [get_rank(item) for item in items]
    ranks = [r for r in ranks if r]
    return sum(ranks) / len(ranks)


def get_rank_or_default(item, default):
    """Use the item’s rank if present, otherwise fallback to default."""
    rank = get_rank(item)
    return rank if rank is not None else default



############## Text-length ##############
def get_text_length(item, PREFIX):
    return len(item.testPrompt(prefix=PREFIX))



def most_common_brands(items, n=40):
    brand_counts = Counter(item.features.get("Brand") for item in items if item.features.get("Brand"))
    return brand_counts.most_common(n)



############## Top-brand flag ##############
def is_top_items_brand(item, top_brands):
    brand = item.features.get("Brand").lower()
    return 1 if brand in top_brands else 0



def get_features(item, avg_weight, avg_rank, top_brands, PREFIX):
    """Bundle all your feature functions into one dict."""
    return {
        "weight": get_weight_or_default(item, avg_weight),
        "rank": get_rank_or_default(item, avg_rank),
        "text_length": get_text_length(item, PREFIX),
        "is_top_electronics": is_top_items_brand(item, top_brands)
    }