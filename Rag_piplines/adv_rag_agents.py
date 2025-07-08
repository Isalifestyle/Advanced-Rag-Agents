
from dotenv import load_dotenv
import hashlib
import os, getpass
from langchain_groq import ChatGroq
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition 
import langsmith
import os
import getpass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import operator
from langgraph.graph import MessagesState
from typing import Annotated
from typing import List
from langgraph.constants import Send
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.vectorstores import Chroma
import chromadb

from langchain.embeddings import HuggingFaceEmbeddings
from langgraph.constants import Send
from langgraph.constants import START, END
from IPython.display import Image
import streamlit as st
chromadb.api.client.SharedSystemClient.clear_system_cache()


# Load environment variables from the .env file
load_dotenv()

# Ensure environment variables are set
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please set {var}: ")



# Debug: Print out the environment variables to ensure they're loaded correctly
print(f"LANGCHAIN_API_KEY: {os.getenv('LANGCHAIN_API_KEY')}")
print(f"LANGSMITH_API_KEY: {os.getenv('LANGSMITH_API_KEY')}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print(f"tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")


# Initialize the LLM (make sure this is correct and that Groq model supports LangSmith tracing)
llm = ChatGroq(api_key="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", model="mistral-saba-24b")



class State(MessagesState):
    query:str
    subqueries: str
    human_feedback_full: str
    retrieval_content:Annotated[list, operator.add]
    subquery:str
    pdf_path: str
    pdf:Annotated[List[str],operator.add]
    chunks:Annotated[List[str],operator.add]
    query_summary:str


def query_expander(state:State):
    
    user_query = state['query']

    human_feedback = state.get('human_feedback_full', "")
    QUERY_REWRITE_FORMAT = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information. Give me 2-3 new quiries. Furthermore, you may be
    supplied with user feeback and if so you need to take this into account to 
    generate queries to the need of the user

    Original query: {original_query}

    Human Feedback: {human_feedback_full}

    Rewritten query:"""
#system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)

    if user_query:
        query_rewrite_prompt = QUERY_REWRITE_FORMAT.format(original_query=user_query, human_feedback_full=human_feedback)
        subqueries = llm.invoke(query_rewrite_prompt)
        subqueries = subqueries.content
    else:
        raise ValueError("Error: Something went wrong")

    st.write(subqueries)


    return {'subqueries':subqueries}


def human_feedback(state: State) -> dict:
    # Display the input field for human feedback using Streamlit
    human_answer = st.text_input(
        "Would you like to escalate this query to an expert? (For critical legal advice, say 'Yes')",

    )
    
    # If the user provided any input, update the state accordingly
    if human_answer.strip():
        state['human_feedback_full'] = 'y' if 'Yes' in human_answer else 'No'

    # Return the next node based on the human feedback
    if state['human_feedback_full'] == 'y':
        return {"next_node": "query_expander"}  # Continue to query expander node
    else:
        return {"next_node": "tavily_search"}  # Proceed to Tavily search node

def decide_next_node(state: dict) -> str:
    """Determines the next node based on the 'next_node' key in the state."""
    return state.get('next_node', '')
from langchain_community.tools.tavily_search import TavilySearchResults



def tavily_search(state:State):
    subquery = state.get('subqueries', ["Are LLMS EVIL?"])

    

    TAVILY_SEARCH_PROMPT = """You are given the following query:
                              {query}
                              Your job is to obtain the best information to answer the set of 
                              questions"""
    TAVILY_API_KEY= os.getenv('TAVILY_API_KEY')

    tavily_query_rewrite = TAVILY_SEARCH_PROMPT.format(query=subquery)


    tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY)
    search_results = tavily_search.invoke(tavily_query_rewrite)



    return {"retrieval_content":[search_results]}

def pdf_loader(state:State):
    pdf_path = state.get("pdf_path",'how__llms__work.pdf')
    pdf_loader = PyMuPDFLoader(pdf_path)
    pdf_pages=[]
    # Extract documents (text)
    documents = pdf_loader.load()
    for doc in documents:
        state['pdf'].append(doc.page_content)  # appending the extracted text to the 'pdf' key in the state
    
    # Returning the updated state
    return {'pdf': state['pdf']}

def split_text(state:State):
    documents = [Document(page_content=text) for text in state['pdf']]
    
    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=10000,
        chunk_overlap=500
    )
    
    # Split the documents into chunks
    chunks = splitter.split_documents(documents)
    
    # Append the chunks to the state['chunks'] list
    for chunk in chunks:
        state['chunks'].append(chunk)
    # Returning the updated state
    return {'chunks': state['chunks']}

def build_vector_store_retrieval(state: State):
    chunks = state.get('chunks', [])  # Use .get() to avoid KeyError if 'chunks' is missing

    # Ensure chunks is a list of Document objects
    documents = [Document(page_content=str(chunk)) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma.from_documents(documents, embedding=embeddings)

    query = state.get('subqueries', '')  # Avoid KeyError if 'subquery' is missing

    search_results = vector_store.similarity_search(query, k=3)

    retrieved_content = [doc.page_content for doc in search_results]

    return {'retrieval_content': retrieved_content}

def summary(state: State):
    all_retrieved_info = state['retrieval_content']
    subqueries = state['subqueries']
    formatted_retrieved_content = "\n".join([doc.page_content if isinstance(doc, Document) else str(doc) for doc in all_retrieved_info])
    summary_prompt = f"""
    You are an AI assistant tasked with summarizing the following retrieved information:

    {formatted_retrieved_content}

    Provide a concise and informative summary of the content and give your references at the end in a organized and neat fashion given the queries: {subqueries}
    """

    response = llm.invoke(summary_prompt)

    st.write("### The following summary was generated:")
    st.write(response.content)



    return {'query_summary': response}



builder = StateGraph(State)

builder.add_node('query_expander', query_expander)
builder.add_node('human_feedback', human_feedback)
builder.add_node('tavily_search', tavily_search)
builder.add_node('pdf_loader', pdf_loader)
builder.add_node('split_text', split_text)
builder.add_node('build_vector_store_retrieval', build_vector_store_retrieval)
builder.add_node('summary', summary)

builder.add_edge(START,'query_expander')
builder.add_edge('query_expander','human_feedback')

builder.add_conditional_edges(
    "human_feedback",
    decide_next_node,   # Reference the function here — no need to define it as a node!
    ["query_expander", "tavily_search"])
builder.add_edge("tavily_search",'pdf_loader')
builder.add_edge("pdf_loader",'split_text')
builder.add_edge("split_text",'build_vector_store_retrieval')
builder.add_edge("build_vector_store_retrieval",'summary')
builder.add_edge("summary",END)


memory = MemorySaver()
compiled_builder = builder.compile(checkpointer=memory)


# query = {'query':'How does RE law work in North Carolina? Do I need to be a broker to buy a house?',}

# thread = {"configurable": {"thread_id": "1"}}


# for event in compiled_builder.stream(query,thread):
        
#     print(event)

# Display the graph (optional)
# display(Image(compiled_builder.get_graph(xray=1).draw_mermaid_png()))


# --------- Streamlit App ------------
st.title("📚 LLM Research Assistant")
st.write("Enter your query below, and I'll provide a detailed response.")

# User Input
user_query = st.text_input("Enter your query:", "")
state = State()
state['human_feedback_full'] = ""  # Initialize with default value



# Handle button click
if st.button("Submit"):
    if user_query.strip():
        # Update the state
        state['query'] = user_query
        state['messages'] = []  # Initialize the messages field
        state['query_summary'] = f"Summary of: {user_query}"  # Example query summary

        thread = {"configurable": {"thread_id": "1"}}

        # Pass the state into the pipeline
        response =compiled_builder.stream(state, thread)
        for event in response:
            if event == 'query_expander':
                st.write(event)
            if event == 'human_feedback':
                st.write(event)
            if event== 'query_summary':
                st.write(event)
