import json
from datasets import load_dataset, Dataset, DatasetDict
from product_price_prediction import ProductExample
from visualize_dataset import summarizeDataset, plotDistribution




def loadDataset():
    """
       Get a dataset from Huggingface:
       The dataset is here: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
    """
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_Appliances", split="full", trust_remote_code=True)
    #print(dataset[0])
    print(f"Number of Appliances: {len(dataset):,}")
    return dataset



def main():
    # ─── 1- Load the dataset and visulaize it ────────────────────────────────────────
    dataset = loadDataset()

    prices, lengths = summarizeDataset(dataset)

    # Length distribution
    plotDistribution(
        lengths,
        title=f"Lengths: avg {sum(lengths) / len(lengths):,.0f}  |  max {max(lengths):,}",
        xlabel="Length (chars)",
        color="lightblue",
        bins=(0, 6000, 100),
    )

    # Price distribution
    plotDistribution(
        prices,
        title=f"Prices: avg ${sum(prices) / len(prices):,.2f}  |  max ${max(prices):,}",
        xlabel="Price ($)",
        color="orange",
        bins=(0, 1000, 10),
    )

    # ─── 2. import our subclass and build examples ──────────────
    examples = []
    for row in dataset:
        try:
            price = float(row["price"])
        except (KeyError, ValueError):
            continue
        if price <= 0:
            continue

        # build our training‐example wrapper
        record = ProductExample(row)

        if record.valid:
            examples.append(record)


    print(f"There are {len(examples):,} valid examples ready for training")

    # peek at your first training prompt + test prompt
    PREFIX = "Price is $"
    print("TRAINING PROMPT:\n", examples[100].prompt, "\n")
    print("TEST   PROMPT:\n", examples[100].testPrompt(prefix=PREFIX), "\n")

    # ─── write to JSONL ─────────────────────────
    with open("train_price.jsonl", "w") as f:
        for record in examples:
            f.write(json.dumps(record.toDict()) + "\n")

    # Plot the distribution of tokens and prices
    tokens = [example.token_count for example in examples]
    plotDistribution(tokens,
             title=f"Token counts: Avg {sum(tokens) / len(tokens):,.1f} and highest {max(tokens):,}\n",
             xlabel="Token counts",
             bins=(0, 300, 10),
             color="green")

    prices = [float(example.raw["price"]) for example in examples if example.valid]
    plotDistribution(prices,
             title=f"Prices: Avg {sum(prices) / len(prices):,.1f} and highest {max(prices):,}\n",
             xlabel="Prices ($)",
             bins=(0, 300, 10),
             color="purple")




if __name__ == "__main__":
    main()
