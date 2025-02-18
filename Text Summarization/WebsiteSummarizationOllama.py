################ Web Page Summarization with Ollama ################

"""
    We do the web page summarization task using Ollama if you prefer avoiding paid APIs.
    Advantages:
      - Open-source, so there are no API fees
      - Keeps all data local to your machine (privacy)

    Disadvantage:
      - Far less powerful than the Frontier Model


    Ollama Installation and Usage:

    1- Install Ollama: Visit ollama.com and install it.
    2- Verify Server is Running:
       - After installation, open your terminal and run "ollama serve".
       - Open your web browser and navigate to http://localhost:11434/.
       - You should see a message confirming that "Ollama is running."

"""

import requests
from bs4 import BeautifulSoup

import ollama
from openai import OpenAI


# Constants
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"



def callOllamaRequest(messages, stream=False):
    """
        Sends a chat request to the Ollama API with the provided messages.

        Parameters:
            messages (list): A list of message dictionaries in the format:
                             {"role": "user", "content": "Your message here"}
            stream (bool): Whether to stream responses. Default is False.

        Returns:
            dict: The API response as a JSON-parsed dictionary, or None if an error occurs.
    """

    paylod = {
        "model": MODEL,
        "messages": messages,
        "stream": stream
    }

    try:
        response = requests.post(OLLAMA_API, json=paylod, headers=HEADERS)
        response.raise_for_status()    # Raise an error for bad status codes
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollam API via requests: {e}")
        return None



def callOllamaPackage(messages):
    """
        Uses the dedicated `ollama` package to send a chat request.

        Parameters:
            messages (list): A list of message dictionaries.

        Returns:
            str: The response message content.
    """
    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]



def callOllamaOpenAI(messages):
    """
        Uses the OpenAI client library to send a chat request to the Ollama API.

        Parameters:
            messages (list): A list of message dictionaries.

        Returns:
            str: The response message content.
    """
    ollama_via_openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollam")

    response = ollama_via_openai.chat.completions.create(model=MODEL, messages=messages)
    # Return the content of the first message in the choices list
    return response.choices[0].message.content



def callOllamaDeepSeek(prompt: str) -> str:
    """
        Calls the Deepseek model via the OpenAI client library to obtain definitions
        for core concepts behind LLMs.

        Parameters:
            prompt (str): The prompt string to send to the model.

        Returns:
            str: The content of the response message from the model.
    """
    # Create the OpenAI client configured for Ollama.
    ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

    messages = [{"role": "user", "content": prompt}]

    response = ollama_via_openai.chat.completions.create(
        model="deepseek-r1:1.5b",
        messages=messages
    )
    return response.choices[0].message.content






if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Describe some of the business applications of Generative AI"}
    ]

    # Method 1: Using requests
    result_requests = callOllamaRequest(messages)
    if result_requests:
        print("Response via requests:")
        print(result_requests)

    # Method 2: Using the ollama package
    result_package = callOllamaPackage(messages)
    print("\nResponse via the ollama package:")
    print(result_package)

    # Method 3: Using the OpenAI client library
    result_openai = callOllamaOpenAI(messages)
    print("\nResponse via the OpenAI client library:")
    print(result_openai)

    # Method 4: Using the Deepseek R1
    prompt = (
        "Please give definitions of some core concepts behind LLMs: "
        "a neural network, attention and the transformer"
    )
    result_deepseek = callOllamaDeepSeek(prompt)
    print("\nResponse via the Deepseek R1:")
    print(result_deepseek)




