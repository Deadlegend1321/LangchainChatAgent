# Hybrid RAG & Web Search Agent

This repository contains the implementation of an intelligent, autonomous **Hybrid Agent** capable of answering queries by dynamically choosing between private indexed knowledge (Retrieval-Augmented Generation) and real-time public web search.

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [Core Architecture & Theory](#core-architecture--theory)
    - [Large Language Models (LLMs) & Chat Models](#1-large-language-models-llms--chat-models)
    - [Embeddings & Vector Databases](#2-embeddings--vector-databases)
    - [The Indexing Pipeline](#3-the-indexing-pipeline)
    - [Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
    - [Web Search Integration](#5-web-search-integration)
    - [The Hybrid Orchestrator](#6-the-hybrid-orchestrator)
3. [System Requirements & Installation](#system-requirements--installation)
4. [How to Run the Application](#how-to-run-the-application)

---

## 🌟 Project Overview

Standard LLMs suffer from two main limitations:
1. **Knowledge Cutoffs**: They cannot answer questions about events occurring after their training phase.
2. **Lack of Private Data**: They do not have access to your personal documents, proprietary wikis, or enterprise databases.

This application builds a **Hybrid Agent** that acts as a unified gateway to solve both problems: 
- **Internal Query?** The agent performs a semantic search over your private/local documents to retrieve accurate, verified facts.
- **External or Real-time Query?** The agent routes the query to a search engine tool (**Tavily API**) to fetch the latest web context.
- **Fallback Control**: It implements logic to search the web if internal documents fail to yield answers.

---

## 🧠 Core Architecture & Theory

### 1. Large Language Models (LLMs) & Chat Models
At the core of the agent is a **Chat Model** (`gpt-4o`), which is an LLM fine-tuned specifically for conversational dialogue and tool interaction. Rather than just predicting the next word, a Chat Model acts as the agent's "brain," reasoning through system prompts to select actions, parse tool returns, and synthesize natural responses.

### 2. Embeddings & Vector Databases
- **Embeddings**: An embeddings model (`text-embedding-3-large`) converts raw text into dense, high-dimensional numerical vectors. This mathematical representation captures the semantic meaning of the words. If two pieces of text have a similar meaning, their vectors will sit close together in vector space.
- **Vector Store**: To query these embeddings efficiently, we use a **Vector Store**. We implement an `InMemoryVectorStore` for fast prototyping, with support for scalable disk-based options like `Chroma` database.

### 3. The Indexing Pipeline
To allow the LLM to read custom websites/files, we process raw content through an **indexing pipeline**:
1. **Document Loading**: Raw HTML is retrieved using `WebBaseLoader`. We filter out boilerplate code (headers, footers, sidebars) using BeautifulSoup4 (`SoupStrainer`) to extract pure semantic text.
2. **Text Splitting**: LLMs have strict context window limitations. We use `RecursiveCharacterTextSplitter` to recursively chunk massive pages into smaller segments of `1000` characters with a `200` character overlap. This preserves continuity across split boundaries.
3. **Vector Ingestion**: Chunks are processed by our embeddings model and loaded into the vector database.

### 4. Retrieval-Augmented Generation (RAG)
RAG addresses LLM hallucinations by changing the LLM's task from "generative guessing" to "closed-book summarization." 
- We wrap our Vector Store in a custom `@tool` named `retrieve_context`.
- When a user asks a question relevant to the documents, the agent invokes this tool, pulls the most relevant text segments, and appends them into its system prompt as ground-truth context.

### 5. Web Search Integration
For questions concerning dynamic, real-time data (e.g., weather, news, current sporting events), the agent leverages **Tavily Search** via `langchain-tavily`. This tool executes web searches, scrapes top results, and delivers a summarized factual payload directly back to the agent's context window.

### 6. The Hybrid Orchestrator
The master agent orchestrates tool routing based on a detailed system prompt defining strict operational parameters:
- **Tool Autonomy**: The agent assesses the query and makes an educated guess on whether to execute `retrieve_context` (for internal content) or `tavily_search` (for external news).
- **Fallback Execution**: If the preferred tool comes back empty or inconclusive, the agent invokes the alternate tool before yielding.
- **Clorification Protocol**: If the query is ambiguous, the agent asks a single clarifying question.
- **Truth Constraint**: The agent is explicitly forbidden from using its pre-trained external knowledge—it must only speak using facts provided by the active tools.

---

## 🛠️ System Requirements & Installation

To run this script locally or inside a PyCharm/VSCode environment, you must install the following packages:

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters beautifulsoup4 langchain-tavily
Ensure you have active API keys for both providers:

OpenAI API Key (for model generation & embeddings)
Tavily API Key (for web search querying)
🚀 How to Run the Application
Create a python script (main.py) or use the notebook to execute the interactive loop. Ensure your environment variables are set:

import os

# Set your credentials
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# Run the chat loop
chat_with_agent(hybrid_agent)
Interactive Commands
Ask about task decomposition to test the RAG internal retrieval.
Ask about current world events or weather to test the Tavily Search tool.
Type bye to exit the application session safely.