import json
from brochureBuilder import Website, connectionAPIKey, getAllRelevantLinks
from brochureBuilder import createBroucher, createStreamBroucher

def main():
    # Initialize and constants
    MODEL, openai = connectionAPIKey()

    url = "https://huggingface.co"
    huggingface = Website(url)
    #print(huggingface.url)

    ####### Step 1: first we find all relevant links from the website #######
    relevant_links = getAllRelevantLinks(url, openai, MODEL)
    print(relevant_links)

    ####### Step 2: make a broucher for the company name #######
    #result = createBroucher("HuggingFace", url, openai, MODEL)
    createStreamBroucher("HuggingFace", url, openai, MODEL)





if __name__ == "__main__":
    main()