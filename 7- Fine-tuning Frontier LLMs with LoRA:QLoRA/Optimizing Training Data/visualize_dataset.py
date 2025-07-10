import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from typing import Sequence, Tuple



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



def summarizeDataset(dataset):
    """ Check our data how many do have a price, and gather the price and the length. """
    prices = []
    lengths = []
    total_prices = 0

    for data in dataset:
        try:
            price = float(data["price"])
            if price > 0:
                prices.append(price)
                total_prices += 1
                contents = data["title"] + str(data["description"]) + str(data["features"]) + str(data["details"])
                lengths.append(len(contents))
        except ValueError as e:
            pass

    print(f"There are {total_prices:,} with prices which is {total_prices/len(dataset)*100:,.1f}%")
    return prices, lengths




def plotDistribution(
        data: Sequence[float],
        title: str,
        xlabel: str,
        color: str = "skyblue",
        bins: Tuple[int, int, int] = (0, 100, 5),  # (start, stop, step)
        figsize: Tuple[int, int] = (15, 6)
):
    """
        Generic 1-D histogram.

        Parameters
        ----------
        data   : list/array of numbers
        title  : plot title (you can embed f-strings)
        xlabel : x-axis label
        color  : bar colour (matplotlib colour spec)
        bins   : (start, stop, step) passed to range() → controls resolution
        figsize: width & height in inches
    """
    start, stop, step = bins
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.hist(data, rwidth=0.7, color=color, bins=range(start, stop, step))
    plt.show()








