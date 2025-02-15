import os
from WebsiteSummarizationAPI import connectionAPIKey, Website, messagesFor


def test(url):
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



def summarize(url):
    # Initialize and constants
    MODEL, openai = connectionAPIKey()

    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messagesFor(website)
    )
    return response.choices[0].message.content



def displaySummary(url):
    summary = summarize(url)
    print(summary)




if __name__ == "__main__":
    url = "https://cnn.com"
    test(url)
    displaySummary(url)
