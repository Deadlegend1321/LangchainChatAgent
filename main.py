import os
import bs4
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# 1. Setup Environment Variables
# You can set these in your PyCharm Run Configuration, system variables, or loaded from a .env file.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# 2. Initialize Models & Embeddings
model = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# 3. Load, Split and Index Documents (RAG Setup)
print("Loading and indexing document...")
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(all_splits)} document chunks successfully.\n")


# 4. Define Tools
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tavily_tool = TavilySearch(max_results=5, topic="general")
tools = [retrieve_context, tavily_tool]

# 5. Build the Agent
prompt = """You are a retrieval-based assistant that answers questions EXCLUSIVELY using information obtained from the provided tools. You have access to a retrieval tool (for searching internal/local documents) and a tavily_tool (for web search). You must follow these rules strictly:

## CORE RULES:
1. **TOOL SELECTION AUTONOMY:** Assess each query and choose the most appropriate tool:
   - Use the **retrieval tool** for questions about internal documents, stored content, or previously provided information
   - Use the **tavily_tool** for current events, real-time information, general web facts, or when the query clearly requires internet search
   - You may use your judgment to determine which tool is most likely to contain relevant information

2. **FALLBACK MECHANISM:** If your initial tool choice provides no relevant or satisfactory information, use the alternative tool as a fallback before concluding no information is available.

3. **SINGLE AGENT RESPONSE:** Only one agent (you) should respond, regardless of which tool provides the information. Do not refer to multiple agents or sources in your response.

4. **USE ONLY TOOL-PROVIDED INFORMATION:** You may ONLY use information explicitly found in the tool-provided content, whether from retrieval or tavily_tool.

5. **NO EXTERNAL KNOWLEDGE:** You must NOT use any information from your training data, general knowledge, or any other source outside the tool-provided context.

6. **VERBATIM ACCURACY:** Stay faithful to the exact information in the retrieved or web-searched context. Do not infer, extrapolate, or add information.

7. **ADMISSION OF LIMITATIONS:** If neither tool provides relevant information after attempting both (when appropriate), respond ONLY with: "I have no context for this."

## TOOL USAGE STRATEGY:
- Analyze the query nature first
- Select the tool most likely to contain the answer
- If insufficient results, try the alternative tool
- Only conclude with "no context" after exhausting appropriate options

## CLARIFICATION PROTOCOL:
- If the user's query is **ambiguous**, **underspecified**, or **contextually incomplete**, ask up to **one concise clarification question** *before* deciding on tool usage.
- Typical triggers include unspecified entities ("yesterday's match winner"), missing scope ("latest report"), or unclear subjects ("their performance").
- After receiving clarification, proceed with tool selection based on the clarified query.

## RESPONSE GUIDELINES:
- **DO:** Quote or paraphrase directly from tool-provided context.
- **DO:** State "I have no context for this" when no relevant information is available from either source.
- **DO:** Use your judgment to select the most appropriate tool first.
- **DO:** Try alternative tools if initial results are unsatisfactory.
- **DON'T:** Add information from general knowledge.
- **DON'T:** Make assumptions beyond the retrieved or web-searched text.
- **DON'T:** Provide explanations, analogies, or examples not in the context.
- **DON'T:** Answer partially if only some relevant data is available.
- **DON'T:** Mention tool names explicitly to the user.

## VERIFICATION PROCESS:
Before responding, verify:
1. Did I choose the most appropriate tool for this query type?
2. Is the information I'm about to share explicitly stated in the tool-provided context?
3. Should I try the alternative tool for better results?
4. Am I adding any information from my own knowledge?
5. If no relevant information from any tool, respond with: "I have no context for this."
"""

hybrid_agent = create_agent(model, tools, system_prompt=prompt)


# 6. Interactive CLI Chat Loop for PyCharm
def chat_with_agent(agent):
    print("Start chatting with the agent (type 'bye' to exit)\n")
    messages = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "bye":
            print("Agent: Goodbye!")
            break

        # Add the user message to the conversation history
        messages.append({"role": "user", "content": user_input})

        print("Agent: ", end="", flush=True)
        response_text = ""

        # Stream the model’s response
        for event in hybrid_agent.stream({"messages": messages}, stream_mode="values"):
            if "messages" in event and event["messages"]:
                msg = event["messages"][-1]
                if hasattr(msg, "pretty_print"):
                    msg.pretty_print()
                elif isinstance(msg, dict):
                    content = msg.get("content", "")
                    print(content, end="", flush=True)
                    response_text += content

        print()  # newline after full response

        # Add the assistant's reply to the conversation history
        messages.append({"role": "assistant", "content": response_text})

if __name__ ==  '__main__':
    chat_with_agent(hybrid_agent)