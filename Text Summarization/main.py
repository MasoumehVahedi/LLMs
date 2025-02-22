import os
from WebsiteSummarizationAPI import connectionAPIKey, Website, messagesFor
from WebsiteSummarizationOllama import callOllamaPackage, callOllamaOpenAI, callOllamaRequest, callOllamaDeepSeek



def test(url):
    #----------------------------------
    #         Testing OpenAI
    # ---------------------------------

    # Initialize and constants
    MODEL, openai = connectionAPIKey()

    # To call to a Frontier model (OpenAI in this case) to get started
    message = "Hello GPT!"
    response = openai.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": message}])
    print(response.choices[0].message.content)

    # The url of the website we want to scrape
    web = Website(url)
    print(web.title)
    print(web.text)

    # ----------------------------------
    #         Testing Ollama
    # ---------------------------------

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




def summarizeWebsiteWithAPI(url):
    # Initialize and constants
    MODEL, openai = connectionAPIKey()

    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messagesFor(website)
    )
    return response.choices[0].message.content



def summarizeWebsiteWithOllama(url):
    website = Website(url)
    # Using the OpenAI client library
    result = callOllamaOpenAI(messages=messagesFor(website))
    return result




def displaySummary(url):
    # Summarization using OpenAI
    print("\nSummary via OpenAI:")
    summary_api = summarizeWebsiteWithAPI(url)
    print(summary_api)

    # Summarization using Ollama
    summary_ollama = summarizeWebsiteWithOllama(url)
    print("\nSummary via the ollama:")
    print(summary_ollama)




def main():
    url = "https://cnn.com"
    # test(url)
    displaySummary(url)




if __name__ == "__main__":
    main()