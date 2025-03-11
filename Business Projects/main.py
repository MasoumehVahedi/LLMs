import json
from brochureBuilder import Website, connectionAPIKey, fetchRelevantLinks



def main():
    # Initialize and constants
    MODEL, openai = connectionAPIKey()

    url = "https://huggingface.co"
    huggingface = Website(url)
    print(huggingface.url)

    relevant_links = fetchRelevantLinks(huggingface.url, openai, MODEL)
    print(json.dumps(relevant_links, indent=2))





if __name__ == "__main__":
    main()