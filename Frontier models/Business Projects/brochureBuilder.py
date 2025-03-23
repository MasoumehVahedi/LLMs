########## Automated Company Brochure Builder ###########

"""
   We develop a tool that automatically creates an informative brochure for any given company using its name and
   primary website. The brochure is designed to attract prospective clients, investors, and potential recruits.

   The technology:
   - Use Open AI API
   - Use one-shot prompting
   - Stream back results and show with formatting

"""

import os
import json
import requests
from typing import List
from openai import OpenAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display



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
    body: str
    links: List[str]

    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "Title NOT Found!"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            # Gather all links
            self.text = ""
        links = [link.get("href") for link in soup.find_all("a")]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"



#--------------------------------------------------------
#            Make a Broucher for a company
#--------------------------------------------------------

###### STEP 1: Use GPT-4o-mini to analyze webpage links and return relevant links in structured JSON format ######
"""
   The model evaluates link relevance and converts relative paths (e.g., "/about") into absolute URLs (e.g., "https://company.com/about").
   This task demonstrates the strength of LLMs in nuanced content comprehension, 
   which would be challenging using traditional web parsing techniques alone ("one-shot prompting" approach).
   Example of one-short prompting:
       "links": [
        {"type": "about page", "url": "https://company.com/about"},
        {"type": "careers page", "url": "https://company.com/careers"}
    ]
   
   Note: A more sophisticated variant, "Structured Outputs," can enforce strict output formats, explored further in Week 8's 
         Autonomous Agentic AI project.
"""

# Define a clear and concise system prompt with structured output expectations.
LINK_SYSTEM_PROMPT = """
You are provided with a list of links from a webpage. Your task is to identify which links are most relevant to include in a company's brochure.
Examples of relevant pages include: About, Company Information, Careers or Jobs.

Respond strictly using the following JSON structure:
{
    "links": [
        {"type": "about page", "url": "https://company.com/about"},
        {"type": "careers page", "url": "https://company.com/careers"}
    ]
}
"""

# Helper function to generate a user prompt dynamically based on a given website
def generateLinksUserPrompt(website):
    prompt = (
        f"Here is the list of links from the website {website.url}. "
        "Decide which of these links are relevant for a company brochure. "
        "Include only relevant pages (e.g., About, Careers, Company Info). "
        "Exclude Terms of Service, Privacy Policy, or email links. "
        "Convert relative links to absolute URLs.\n\n"
        "Links:\n"
    )
    prompt += "\n".join(website.links)
    return prompt


# Core function to interact with the GPT-4o-mini model to retrieve relevant links
def fetchRelevantLinks(url, openai, MODEL):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": LINK_SYSTEM_PROMPT},
            {"role": "user", "content": generateLinksUserPrompt(website)}
        ],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    return json.loads(content)


def getAllRelevantLinks(url, openai, MODEL):
    result = "Landing Page\n"
    result += Website(url).get_contents()
    relevant_links = fetchRelevantLinks(url, openai, MODEL)
    print(f"Found Links: {relevant_links}")
    for link in relevant_links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result



###### STEP 2: Use the information from step 1 to make a brocher ######
# With changing system_prompt words instead of saying "prospective customers, investors and recruits", if we say
# "short humorous, entertaining, jokey brochure", everything will be changed and the content will be different.
system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

def getBroucherUserPrompt(company_name, url, openai, MODEL):
    user_prompt = f"We are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += getAllRelevantLinks(url, openai, MODEL)
    user_prompt = user_prompt[:5_000]  # Truncate if more than 5,000 characters
    return user_prompt


def createBroucher(company_name, url, openai, MODEL):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": getBroucherUserPrompt(company_name, url, openai, MODEL)}
        ],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    return json.loads(content)



def createStreamBroucher(company_name, url, openai, MODEL):
    """ This is fancy format so that the results stream back from OpenAI. """
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": getBroucherUserPrompt(company_name, url, openai, MODEL)}
        ],
        stream = True
    )
    response = ""
    #display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        response = response.replace("```","").replace("markdown", "")
        #update_display(Markdown(response), display_id=display_handle.display_id)
        print(response, end="\r", flush=True)





