import pandas as pd
from features import get_features


def list_to_dataframe(items, feature_fn):
    """
        A utility function to convert our features into a pandas dataframe.
        Arg:
            - items:    list of ProductExample (with .price & .features)
            - feature_fn: function(item) -> dict of numerical features
    """
    features = [feature_fn(item) for item in items]
    df = pd.DataFrame(features)
    df["price"] = [item.price for item in items]
    return df