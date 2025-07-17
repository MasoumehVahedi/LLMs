import pandas as pd
import csv



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



def export_human_input(test_file, filename="human_input.csv", prefix="Price is $"):
    """
        Write the first 250 test prompts to a CSV for manual labeling.

        Each row will be [prompt_without_answer, 0].
        Human annotators can replace the 0’s with their price estimates.
    """
    with open(filename, "w") as csvfile:
      writer = csv.writer(csvfile)
      for test in test_file[:250]:
          writer.writerow([test.testPrompt(prefix=prefix), 0])


def load_human_predictions(filename="human_output.csv"):
    """
        Read back the CSV after humans fill in their estimates.

        Expects each row to be [prompt, human_price].
        Returns a list of floats.
    """
    human_preds = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            human_preds.append(float(row[1]))
    return human_preds


def make_human_price(test, human_preds):
    """
        Return a predictor(item) function that looks up the human’s price
        by matching the item’s index in test_items.
    """
    def price(item):
        idx = test.index(item)
        return human_preds[idx]

    price.__name__ = "human_price"
    return price

