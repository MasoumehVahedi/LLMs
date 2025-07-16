"""
    We are now evaluating state‑of‑the‑art LLMs (“Frontier Models”) on our price‑prediction task.

    Unlike our traditional ML baselines (trained on 136k examples), we’re not fine‑tuning these massive LLMs,
    they only see the test inputs at query time.

    However, because their pretraining data is so vast, it may already include (or “memorize”) many of our test items,
    introducing test‑set leakage that can artificially boost their apparent performance.

"""


import os
import re
from openai import OpenAI

# Utilities for querying a large frontier LLM (e.g. gpt-40-mini)
# on our price-prediction task.

# Initialize an OpenAI client (make sure you've set OPENAI_API_KEY in your environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def prepare_messages(item):
    """
        Build a chat message sequence for a Frontier model to estimate an item's price.

        Args:
            item:  Any object with a .test_prompt() method that returns the prompt text.
        Returns:
            List of messages conforming to the OpenAI chat API format.
    """
    system_message = "You are a price estimate assistant. Given a product description, reply only with the numeric price, no commentary."
    # When we train our own models, we'll need to make the problem as easy as possible,
    # but a Frontier model needs no such simplification. We do not need such guardrail.
    user_message = item.testPrompt(prefix="Price is $").replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Price is $"}
    ]


def parse_number(text: str) -> str:
    """
        Extract the first integer or decimal number from text and return it.
        Returns 0.0 if no number is found.
    """
    clean = text.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", clean)
    return float(match.group()) if match else 0



def frontier_price(item,
                   model: str = "gpt-4o-mini",
                   max_tokens: int = 10,
                   seed: int = 42,
                   temperature: float = 0.0
                   ):

    """
        Query a Frontier model to estimate the price of an item.

        Args:
            item: Any object with a .testPrompt() method that returns the prompt text.
        Returns:
            Estimated price as a float.
    """
    messages = prepare_messages(item)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed
    )
    content = response.choices[0].message.content
    return parse_number(content)  # Extract the numeric price from the response



def make_frontier_pricer(model_name: str,
                         max_tokens: int = 5,
                         seed: int = 42,
                         temperature: float = 0.0):
    """
    Return a predictor(item) that calls `frontier_price` with our chosen model.
    """
    def pricer(item):
        return frontier_price(
            item,
            model=model_name,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature
        )
    # give it a nice __name__ for Tester output
    pricer.__name__ = model_name.replace("-", "_")
    return pricer
