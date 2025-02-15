############### Web Page Summarization ###############
"""
    What what business problem are we actually going To solve?

    - We're going to write a program which is going to be able to look at any web page on the internet, scrape
      the contents of the web page and then summarize it and present back a short summary of that web page.
    - We can think of it like we're building our own little web browser, which is like a summarizing web browser.

"""

import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup



#------------------------------------
#     Call to a Frontier model
#------------------------------------

def connectionAPIKey():
    # Connecting to OpenAI
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key and api_key.startswith("sk-proj-") and len(api_key) > 0:
        print("API key was found!")
    else:
        print("There is a problem to find the API key!")

    # To connect OpenAI
    MODEL = "gpt-4o-mini"
    openai = OpenAI()

    return MODEL, openai


#---------------------------------------
#       Represent a Web Page
#---------------------------------------

# Some websites need to use proper headers when fetching them:
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.92 Safari/537.36"
}

class Website:
    """ A class represents scraped websites with links """

    url: str
    text: str
    title: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")   # To parse a web page
        self.title = soup.title.string if soup.title else "Title NOT Found!"   # To pluck out the title of the web page
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


#-------------------------------
#       Types of prompts
#-------------------------------

""" 
   Frontier models such as GPT4o have been trained to get instructions in a specific way.
   They are expecting to get:
   
   - A system prompt: that instructs them on the task at hand and the tone to adopt.
   - A user prompt: the conversation starter that they should reply to.
   
   IMPORTANT NOTE:
   The API from OpenAI expects to receive messages in a particular structure. Many of the other APIs share this structure:

    [
        {"role": "system", "content": "system message goes here"},
        {"role": "user", "content": "user message goes here"}
    ]

   example:
           messages = [
                        {"role": "system", "content": "You are a snarky assistant"},
                        {"role": "user", "content": "What is 2 + 2?"}
                      ]
"""

# An example of system prompt
system_prompt = "As an assistant, your role is to examine the contents of a " \
                "website and deliver a short summary, omitting text associated with navigation. " \
                "Your response should be formatted in markdown."

# An example of user prompt
def userPromptFor(website):
    """ A function that writes a User Prompt that asks for summaries of websites """
    user_prompt = f"We are searching at a website titled {website.title}"
    user_prompt += "\nThis website contains the following information: " \
                   "Please generate a short markdown summary of this website. " \
                   "If there are announcements, please summarize those as well.\n\n"
    user_prompt += website.text
    return user_prompt



def messagesFor(website):
    """ To build useful messages for GPT-4o-mini """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": userPromptFor(website)}
    ]







