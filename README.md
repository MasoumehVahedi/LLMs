
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

## 📖 Key Concepts & Workflow Overview:

## 1. Metric Definitions

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

## 2. Vector Embeddings & Encoding

High-dimensional vectors are a powerful way to **mathematically represent the “meaning”** of almost any piece of data—be it a single character or token, a word, a sentence or paragraph, an entire document, or even something abstract (for example, candidate profiles, job descriptions, product features, etc.).  


### Auto-Regressive vs Auto-Encoding Models

- **Auto-Regressive LLMs**  
Generate text by predicting each next token from the sequence of previous tokens (e.g., GPT-style models).  
- **Auto-Encoding LLMs**  
Consume the entire input at once and produce outputs like classifications or embeddings. These models are the backbone of most embedding services: you feed in text, and they return a fixed-length vector.  

**Common embedding models**  
- BERT (Google)  
- OpenAI Embedding API (e.g., `text-embedding-ada-002`)  

### Why Embeddings Matter

By converting text or other data into vectors that “understand” meaning:

1. **Semantic Search & Retrieval**  
 Quickly find the most relevant documents or passages via nearest-neighbor lookup in vector space.  
2. **Retrieval-Augmented Generation (RAG)**  
 Retrieve contextually relevant snippets and prepend them to our LLM prompt—grounding responses in real data and improving factual accuracy.  

## 🧩 3. The Big Idea Behind RAG

Retrieval-Augmented Generation (RAG) marries the strengths of vector search with large language models to ground your answers in real data. Here’s how it works:

1. **User Question → Vector**  
   - An _encoding_ model turns the incoming question (“Who is Anna Depp?”) into a fixed-length vector that captures its meaning.

2. **Vector Store Lookup**  
   - We maintain a **vector datastore**—every document or knowledge snippet has already been encoded into its own vector. \ 
   - We perform a nearest-neighbor search to find which stored vectors lie closest to the question vector in semantic space.

3. **Fetch Relevant Text**  
   - For each retrieved vector, we pull back the original text snippet it represents (e.g. the Wikipedia paragraph on Anna Depp).

4. **Augment the Prompt**  
   - We prepend the fetched snippets to the user’s question, forming a context-rich prompt.

5. **LLM Generation**  
   - This augmented prompt goes into an auto-regressive LLM (e.g. GPT). The model “sees” both the question and the relevant facts and generates a grounded, accurate answer.

---

### Why RAG?

- **Improved Accuracy**  
  The LLM can’t hallucinate about facts it hasn’t seen—every answer is backed by real documents.  
- **Scalability**  
  We can index millions of pages in a vector database and still retrieve relevant context in milliseconds.  
- **Flexibility**  
  Swap in different encoders or vector stores (FAISS, Annoy, Pinecone, etc.) without changing your LLM pipeline.

> **Flow Diagram (conceptual)**  
> ```
> [User Question]  
>      ↓  
> [Encoding LM: “Who is Anna Depp?” → 𝕧ᵩ]  
>      ↓  
> [Vector Datastore: find docs whose vectors ≈ 𝕧ᵩ]  
>      ↓  
> [Retrieve top-k text snippets]  
>      ↓  
> [Augment Prompt: snippets + question]  
>      ↓  
> [LLM Generation → Answer]  
> ```

## 4. Introducing Chroma

Chroma is the open-source “AI application database” – batteries included. It brings together:

- **Embeddings & Vector Search**  
  Store and query high-dimensional vectors for semantic lookup.  
- **Document & Full-Text Search**  
  Keep our raw text alongside embeddings and fall back to keyword search when you need it.  
- **Metadata Filtering & Multi-Modal Support**  
  Tag and filter records, or index images, JSON, and more in one unified store.  

This is what we'll be using to store our vectors.


## 5. Finding & Curating Datasets

Before you can encode and index text, you need a corpus. Common sources include:

1. **Your Own Proprietary Data**  
   – Extract text from internal wikis, customer support logs, product specs, or other domain-specific repositories you already own.

2. **Kaggle**  
   – A rich ecosystem of public datasets.

3. **Hugging Face Datasets**  
   – A centralized hub of hundreds of ready-to-use NLP and multimodal datasets with easy Python integration (`datasets` library).

4. **Synthetic Data**  
   – Generate custom examples (e.g. via prompting a large model) when you need edge cases or to fill gaps in your real-world data.

5. **Specialist Data Providers**  
   – Companies like Scale AI or Appen will curate, annotate, and validate datasets tailored to your exact requirements (e.g. entity-level annotations, multilingual corpora, etc.).


## 6. Digging into the Data  
A six-step data preparation loop—each with its own one-sentence purpose:  
1. **Investigate:** explore formats, size, and quirks of your raw data.  
2. **Parse:** convert inputs into structured records (JSON, CSV, DB).  
3. **Visualize:** chart distributions and outliers to uncover patterns.  
4. **Assess Data Quality:** measure missing values, bias, and consistency.  
5. **Curate:** clean, filter, and enrich to craft the ideal training set.  
6. **Save:** persist your final dataset (local files, HF Hub, etc.).  



## 7. Token Limits & Cost  
Every encoder/LLM has a max context length (in tokens). By choosing a fixed budget (e.g. 2K–4K tokens) and chunking your documents accordingly, you:  
- **Predict resource needs** for GPU training (memory, batch size)  
- **Cap API spend** on hosted models (billed per token)  
- **Ensure fair comparison** between frontier and open-source models  


## 8. 5-Step LLM Strategy  
To take an LLM from concept to production:  
1. **Understand**: define business goals, success metrics, data scope, and non-functional requirements.  
2. **Prepare**: research existing solutions, compare models (context length, cost, license), and curate your data.  
3. **Select**: pick prompting, RAG, or fine-tuning based on accuracy needs, dataset size, and budget.  
4. **Customize**: apply your chosen technique—prompt engineering for quick wins, RAG for grounded accuracy, or fine-tuning for specialist performance.  
5. **Productionize**: wrap the model in a robust API, deploy with monitoring/security, measure against business metrics, and set up a continuous retrain loop.  


## 9. Technique Cheat-Sheet  
| Technique  | Best For                                                               | Pros                                       | Cons                                             |
|------------|------------------------------------------------------------------------|--------------------------------------------|--------------------------------------------------|
| Prompting  | Quick prototype on a frontier API                                      | Fast, low cost, immediate improvement      | Limited by context length, diminishing returns   |
| RAG        | High factual accuracy without full model training, with existing KB    | Accurate with low data needs, scalable     | Requires maintained vector store, up-to-date data |
| Fine-tuning| Specialized tasks, large high-quality datasets, nuanced behavior       | Deep expertise, consistent style, efficient| High data needs, training cost, catastrophic forgetting |

Use this as your roadmap: iterate through **Understand → Prepare → Select → Customize → Productionize**, and you’ll have a repeatable, scalable LLM engineering process. 



## Reference

