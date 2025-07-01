""" A minimal Retrieval-Augmented Generation pipeline.

* Loads every Markdown file under `knowledge_base/`
  (sub-folders become metadata: doc_type=products | employees | …)
* Splits docs into chunks (size & overlap configurable)
* Embeds chunks with OpenAIEmbeddings  ❱ you can swap to HF easily
* Stores embeddings in a local Chroma vector store
* Answers questions with a LLM chain (e.g., GPT-4)

No extra infra, no API gymnastics: run, ask, done.
"""


import os
import glob
import gradio as gr
from dotenv import load_dotenv
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from typing import Optional, Literal

# imports for langchain and Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler



class RAGAssistant:
    """
        Build a conversational RAG bot in one line:

            rag = RAGAssistant(folder_root="knowledge_base")
            rag.launch_gradio()

        Parameters
        ----------
        folder_root   : root folder with sub-folders of .md files
        chunk_size    : characters per chunk
        chunk_overlap : overlap between chunks
        k             : # of chunks to send to the prompt
        embed_backend : "hf" | "openai"
        model_name    : OpenAI chat model if you choose openai backend
    """
    def __init__(self,
                 folder_root: str = "knowledge-base",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 k: int = 4,
                 embed_backend: Literal["hf", "openai"] = "hf",
                 db_name: str = "vector_db",
                 openai_api_key: Optional[str] = "None",
                 use_openai_embed: bool = False,
                 debug: bool = True,
                 model_name: str = "gpt-4o-mini"):
        self.folder_root = folder_root
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_name = db_name
        self.embed_backend = embed_backend,
        self.model_name = model_name
        self.k = k            # chunks sent to the prompt
        self.debug = debug    # StdOutCallbackHandler


        # ─── 1.  Embeddings ───────────────────────────────────────────
        if use_openai_embed:
            from langchain.embeddings import OpenAIEmbeddings
            os.environ["OPENAI_API_KEY"] = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ─── 2.  LLM  (chat model) ────────────────────────────────────
        self.openai_api_key = self.connectionAPIKey()
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.llm = ChatOpenAI(temperature=0.7, model_name=self.model_name, openai_api_key=self.openai_api_key)
        # Alternative - if we would like to use Ollama locally
        # self.llm = ChatOpenAI(temperature=0.7, model_name='llama3.2', base_url='http://localhost:11434/v1', api_key='ollama')

        # ─── 3.  Build vector store: load → chunk → embed ───────────────────────────────────
        self.vectorstore = self.buildVectorstore()

        # ─── 4.  chain & memory  ────────────────────────────
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        callbacks = [StdOutCallbackHandler()] if self.debug else None
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            callbacks=callbacks
        )

    @staticmethod
    def connectionAPIKey(env_var: str = "OPENAI_API_KEY") -> str | None:
        """
        Load env file, fetch key, validate the format.
        Returns the key (or None) so you can pass it straight to ChatOpenAI.
        """
        load_dotenv(override=True)  # reads .env
        api_key = os.getenv(env_var)

        if api_key and api_key.startswith("sk-") and len(api_key) > 20:
            print("OpenAI API key found.")
            return api_key
        print("Unable to find a valid OpenAI key.")
        return None

    # ------------------------------------------------------------------
    # 1. Load and tag docs
    # ------------------------------------------------------------------
    def loadDocuments(self):
        """ Read in documents using LangChain's loaders
            Take everything in all the sub-folders of the knowledge-base folder.
        """
        folders = glob.glob(os.path.join(self.folder_root, "*"))
        documents = []

        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            for doc in loader.load():
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)

        return documents


    # ------------------------------------------------------------------
    # 2. Split text into chunk → embed → store
    # ------------------------------------------------------------------
    def buildVectorstore(self):
        """ 1- We split the text into chunks.
            2- We map each chunk of text into a Vector that represents the meaning of the text, known as an embedding.
            3- Create our Chroma vectorstore.
        """
        documents = self.loadDocuments()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
        print(f"Document types found: {', '.join(doc_types)}")

        # Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, embedding_function=self.embeddings).delete_collection()
        # build vector DB from chunks
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_name
        )

        print(f"Vectorstore created with {vectorstore._collection.count()} documents")
        return vectorstore


    # ----------------------------------------------------------
    # 3. Public API
    # ----------------------------------------------------------
    def chat(self, query: str) -> str:
        """Ask a question from our knowledge-base files."""
        response = self.conversation_chain.invoke({"question": query})
        return response["answer"]

    # ----------------------------------------------------------
    # 4. Gradio chat UI
    # ----------------------------------------------------------
    def launchGradio(self, title="Knowledge Worker") -> None:
        """Open a Gradio chat UI in the browser."""

        def chat_fn(q, hist):  # hist is ignored (memory lives in chain)
            return self.chat(q)

        gr.ChatInterface(chat_fn, title=title).launch(inbrowser=True)


    # ------------------------------------------------------------------
    # 5. Visualizing the Vector Store
    # ------------------------------------------------------------------
    def visualise(self, dims: int = 2) -> None:
        """TSNE plot of all vectors (2‑D or 3‑D)."""
        if self.vectorstore is None:
            raise RuntimeError("No vectors – run ingest() first")
        res = self.vectorstore._collection.get(include=["embeddings", "documents", "metadatas"])
        vecs = np.array(res["embeddings"])
        labels = [m["doc_type"] for m in res["metadatas"]]
        docs = res["documents"]
        tsne = TSNE(n_components=dims, random_state=42).fit_transform(vecs)

        color_map = {k: c for k, c in zip(sorted(set(labels)), ["blue", "green", "red", "orange", "purple", "cyan"])}
        colors = [color_map[l] for l in labels]

        if dims == 2:
            fig = go.Figure(data=go.Scatter(
                x=tsne[:, 0], y=tsne[:, 1], mode="markers",
                marker=dict(size=6, color=colors),
                text=[f"{t}<br>{d[:100]}…" for t, d in zip(labels, docs)],
                hoverinfo="text"))
        else:
            fig = go.Figure(data=go.Scatter3d(
                x=tsne[:, 0], y=tsne[:, 1], z=tsne[:, 2], mode="markers",
                marker=dict(size=4, color=colors, opacity=0.8),
                text=[f"{t}<br>{d[:100]}…" for t, d in zip(labels, docs)],
                hoverinfo="text"))
        fig.update_layout(title="Vector space (TSNE)")
        fig.show()





if __name__ == "__main__":
    bot = RAGAssistant(folder_root="knowledge-base",
                       embed_backend="hf",  # no cost
                       debug=True,
                       k=25)

    print(bot.chat("Who received the prestigious IIOTY award in 2023?"))  # if it says "I do not know", it is wrong because we have this info in the files.
    bot.launchGradio()  # opens a chat window

    # visualize vector embedding
    bot.visualise(dims=3)








