import matplotlib.pyplot as plt
from typing import Sequence, Tuple




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



def plotBar(
        labels: Sequence[str],
        counts: Sequence[int],
        title: str = "",
        xlabel: str = "",
        ylabel: str = "Count",
        color: str = "skyblue",
        figsize: Tuple[int, int] = (15, 6),
        rotation: int = 0
):
    """
        Plot a bar chart with counts on top of each bar.
    """
    plt.figure(figsize=figsize)
    bars = plt.bar(labels, counts, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")

    # add value labels
    for i, bar in enumerate(bars):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h:,}",
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.show()



def plotDonut(
    labels: Sequence[str],
    values: Sequence[float],
    title: str = "",
    figsize: tuple[int, int] = (12, 10),
    autopct: str = "%1.0f%%",
    startangle: int = 90,
    hole_radius: float = 0.70,
):
    """
    Draws a donut chart (pie with a hole) for the given categories.

    Args:
        labels:   sequence of category names
        values:   sequence of numeric values (same length as labels)
        title:    chart title
        figsize:  figure size in inches
        autopct:  percent format string
        startangle: where to start first slice (0-360)
        hole_radius: radius of the central hole (0.0–1.0)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(
        values,
        labels=labels,
        autopct=autopct,
        startangle=startangle,
    )
    # draw center circle to make it a donut
    centre = plt.Circle((0, 0), hole_radius, color="white")
    ax.add_artist(centre)
    ax.set_title(title)
    ax.axis("equal")  # keep it circular
    plt.show()









