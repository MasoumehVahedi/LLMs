import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
from collections import Counter, defaultdict
from datasets import load_dataset, Dataset, DatasetDict

from visualize_dataset import plotDistribution, plotBar, plotDonut
from balance_training_sample import buildTrainingSample
from product_price_prediction import ProductExample
from dataset_loader import DatasetLoader




def connectionAPI():
    """
       Connection to API
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")

    hf_token = os.getenv("HF_TOKEN")
    return openai_api_key, hf_token


def report(record: ProductExample):
    prompt = record.prompt
    # Get the tokenizer that was passed in when this Example was created
    tok = record.tokenizer

    # Encode the prompt into token IDs
    token_ids = tok.encode(prompt)

    print("\n=== Prompt ===")
    print(prompt)

    print("\n=== Last 10 token IDs ===")
    print(token_ids[-10:])

    print("\n=== Decoded last 10 tokens ===")
    print(tok.batch_decode(token_ids[-10:]))



def dataCuration():
    """
       Efficient and fast way to load a large dataset.
    """

    # Load in one dataset
    dataset_name = "Appliances"
    records = DatasetLoader(dataset_name).load()
    print(records[1].prompt)

    # -----------------------------------------
    #        Optimizing Large Dataset
    # -----------------------------------------
    # Now, we SCALE UP all datasets of all records (very large dataset)
    dataset_names = [
        "Video_Games",
        "Musical_Instruments",
        "Appliances",
        "Industrial_and_Scientific",
        "Software"
    ]

    all_records = []
    for dataset_name in dataset_names:
        loader = DatasetLoader(dataset_name)
        all_records.extend(loader.load())

    print(f"A grand total of {len(all_records):,} records")

    # Plot the distribution of tokens and prices
    tokens = [example.token_count for example in all_records]
    plotDistribution(tokens,
                     title=f"Token counts: Avg {sum(tokens) / len(tokens):,.1f} and highest {max(tokens):,}\n",
                     xlabel="Token counts",
                     bins=(0, 300, 10),
                     color="green")

    prices = [float(example.price) for example in all_records]
    plotDistribution(prices,
                     title=f"Prices: Avg {sum(prices) / len(prices):,.1f} and highest {max(prices):,}\n",
                     xlabel="Prices ($)",
                     bins=(0, 300, 10),  # bins=(0, 10000, 10),
                     color="purple")

    # The bar chart to show us how many we have in each of the different categories of data of product.
    category_counts = Counter(record.category for record in all_records)
    cats = list(category_counts.keys())
    cnts = [category_counts[c] for c in cats]

    plotBar(
        labels=cats,
        counts=cnts,
        title="How many in each category",
        xlabel="Categories",
        color="goldenrod",
        rotation=30,
    )

    # ----------------------------------------------------------
    #        Create a Balanced Dataset for LLM Training
    # ----------------------------------------------------------
    slots = defaultdict(list)
    for example in all_records:
        slots[round(example.price)].append(example)

    sample = buildTrainingSample(slots, dataset_name="Industrial_and_Scientific")
    print(f"There are {len(sample):,} items in the sample")

    # Plot the distribution of prices in sample
    prices = [float(example.price) for example in sample]
    plotDistribution(prices,
                     title=f"Avg {sum(prices) / len(prices):.2f} and highest {max(prices):,.2f}\n",
                     xlabel="Prices ($)",
                     bins=(0, 1000, 10),  # bins=(0, 10000, 10),
                     color="darkblue")

    category_counts = Counter(record.category for record in sample)
    categories = list(category_counts.keys())
    counts = [category_counts[c] for c in categories]

    plotBar(
        labels=cats,
        counts=counts,
        title="How many in each category",
        xlabel="Categories",
        color="lightgreen",
        rotation=30,
    )

    plotDonut(
        labels=categories,
        values=counts,
        title="Distribution of Records by Category"
    )

    #-----------------------------------------
    #        Analysing Correlations
    #-----------------------------------------
    # How does the price vary with the character count of the prompt?
    # Let’s measure the correlation to see whether longer prompts actually tend to produce higher prices.
    size_prompt = [len(record.prompt) for record in sample]
    prices = [example.price for example in sample]

    # Scatter plot of prompt-length vs. price
    plt.figure(figsize=(15, 8))
    plt.scatter(size_prompt, prices, s=0.2, color="red")
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.title('Is there a simple correlation?')
    plt.show()

    # To check prompt and tokenizer
    report(sample[39800])



