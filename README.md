
# Large Language Models

This project highlights some of the most advanced large language models (LLMs) and their corresponding end-user applications. 

### Some important Frontier models and their end-user:

1- OpenAI
Models: GPT, O1
Chat: ChatGPT

2- Anthropic
Models: Claude
Chat: Claude

3- Google
Models: Gemini
Chat: Gemini Advance

3- Cohere
Models: Command R+
Chat: Command R+

4- Meta
Models: Llama
Chat: meta.ai

5- Perplexity
Models: Perplexity
Search: Perplexity

## Building AI Agent Framework

### Agentic AI: Smarter Problem-Solving

Agentic AI helps AI agents tackle problems efficiently. Here's how:

- **Break tasks into steps** → Different AI models handle different parts.
- **Use tools** → AI can access extra tools for better results.
- **Collaborate** → AI agents work together in an environment.
- **Plan ahead** → One AI organizes tasks for the others.
- **Think independently** → AI remembers and acts beyond simple replies.

## Comparing LLMs to select for a project

Compare the following features of an LLM.

1- Basics 1:

- **Open-source or closed**
- **Release date and knowledge cut-off**
- **Parameters**
- **Training tokens**
- **Context length**

2- Basics 2:

- **Inference cost** -> API charge, Subscription or Runtime compute
- **Training cost**
- **Build cost** -> How much work will it be for you to create this solution.
- **Time to Market** -> How long does it take to build the LLM
- **Rate limits** -> run into some limits on how frequently you can call them. This is typically the case for subscription plans.
- **Speed** -> How quickly can generate a whole response?
- **Latency** -> How quickly does it first start responding with each token?
- **License** -> Whether we are dealing with open source or closed source to be allowed to use it.

3- The Chincilla Scaling Law:

The number of parameters is roughly proportional to the size of our training data to the number of training tokens. \
Let's say it's an 8 billion parameter model, and we get to the point where we start to see that we're getting diminishing returns. Adding in more training data isn't significantly affecting the model.

How many more parameters do I need given extra training data? \
Answer: if we were then to double the amount of training data from that that point of diminishing returns, we would need double the number of weights you'd need to go from 8 billion to 16 billion parameters to be able to consume twice the training data and learn from it in an effective way.

How much more training data am I going to need to be able to to take advantage of that? \
Answer is: we would you would roughly need to double the size of our training data set.

## LLM Benchmarks

**Hugging Face Open LLM leaderboard**, which is where we can go to compare open source models against a number of different benchmarks and and key attributes.
Visit this [link] (https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/).

## Six Essential Leaderboards

Add these to our bookmarks, they’re great resources for selecting LLMs:

1- HuggingFace Open LLM – new and old versions \
2- HuggingFace BigCode \
3- HuggingFace LLM Perf \
4- HuggingFace Others – medical and language-specific \
5- Vellum – includes API cost and context window info \
6- SEAL – assesses expert-level skills

## LLM Leaderboard & Evaluation Resources

- **LMSYS Chatbot Arena** -> Helps determine which model is better at instruction-following chats.

1- Compare frontier and open-source models directly. \
2- Blind human evaluations based on head-to-head comparisons. \
3- Models are rated using an ELO system. \

## How to evaluate the performance of a Gen AI solution?
*This is perhaps the single most important question you will face*

| Model-centric or Technical Metrics     | Business-centric or Outcome Metrics     |
|----------------------------------------|-----------------------------------------|
| **Loss** (e.g., cross-entropy loss)    | **KPIs tied to business objectives**    |
| **Perplexity**                         | - **ROI**                               |
| **Accuracy**                           | - **Improvements in time, cost or resources** |
| **Precision, Recall, F1**              | - **Customer satisfaction**             |
| **AUC-ROC**                            | - **Benchmark comparisons**             |
| *Easiest to optimize with*             | *Most tangible impact*                  |


## Metric Definitions

- **Cross-Entropy Loss**  
  The negative log‐probability of the true next token. It’s a core training objective—lower is better.

- **Perplexity**  
  Defined as e to the power of cross-entropy loss, it measures how “surprised” the model is.  
  A perplexity of 1 means perfect prediction; higher values indicate more uncertainty.

> **Tip:** We need to use them in concert. One allows us to optimize our model to fine tune our model to to demonstrate its its fast performance. And the other of them is what we use to ultimately prove the business impact behind our solution.

## Motivating RAG: Retrieval Augmented Generation
### The problem with static prompts
Even the best prompt can only “see” what we give it up front.  
- We ask “What’s our current product lineup?”  
- 🤖 The model guesses based on training data, and may be out-of-date or incomplete.

### A simple RAG workflow
1. **Query a live Knowledge Base**  
   – e.g. a company wiki, product catalog, or database of FAQs  
2. **Retrieve the most relevant passages**  
   – “List all active insurance products”  
3. **Stitch those snippets into your prompt**  
4. **Send the enriched prompt to the LLM**  
5. **Return an accurate, up-to-date answer**

### Real-world example
> **User**: “Which policies cover flood damage?”  
> **RAG step 1**: Search Knowledge Base for “flood” → finds two policy docs  
> **RAG step 2**: Prepend:  
> ```
> [Policy A: covers flood up to $10k]  
> [Policy B: excludes flood coverage in coastal zones]
> ```  
> **Prompt to LLM**:  
> ```
> We have two policies: A covers flood up to $10k; B excludes flood in coastal zones.  
> Q: Which policies include flood damage coverage?
> ```  
> **LLM Answer**: “Only Policy A covers flood damage (up to \$10,000).”

### Why it matters
- **Up-to-date**: Always reflects our latest data  
- **Focused**: Filters out irrelevant info  
- **Explainable**: We can show exactly which sources we used  
- **Scalable**: Swap in any Knowledge Base—wikis, SQL databases, vector stores  

With RAG, our app goes from “best guess” to “best answer.”



## Reference

