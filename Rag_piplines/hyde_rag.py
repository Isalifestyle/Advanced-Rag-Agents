from dotenv import load_dotenv
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
import streamlit as st
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
from langgraph.constants import START, END
from IPython.display import Image

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
    hypothetical_response:str
    pdf_file_path:str
    official_response:str
    chunks:Annotated[List[str], operator.add]
    pdf:Annotated[List[str],operator.add]
    query_summary:str
    human_feedback:str
    next_node: str


def hypo_response(state:State):
    query = state.get('query','')
    human_feedback = state.get('human_feedback','')
    PROMPT = """You are tasked with giving a client the most accurate information possible. You must not comdemn the use of hallucination. 
                Here is the query: {query}
                
                There also might be human feedback to consider. If so then take into account this heavily and adjust response accordingly
                
                Human Feedback: {human_feedback}

                """

    PROMPT_REWRITE_FORMAT = PROMPT.format(query=query, human_feedback = human_feedback)

    hypo_response = llm.invoke(PROMPT_REWRITE_FORMAT)

    st.write("Hypothetical Response:")
    st.write(hypo_response.content)


    return {'hypothetical_response':hypo_response}

def pdf_loader(state:State):
    pdf_path = state.get("pdf_file_path","./RE_law_commision.pdf")

    pdf_loader = PyMuPDFLoader(pdf_path)

    documents = pdf_loader.load()
    for doc in documents:
        state['pdf'].append(doc.page_content)

    return {"pdf":state['pdf']}


def split_text(state:State):
    documents = [Document(page_content=text) for text in state['pdf']]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=10000,
        chunk_overlap=500
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks: 
        state['chunks'].append(chunk)

    return {'chunks': state['chunks']}


def build_vector_store_retrieval(state:State):
    chunks = state.get('chunks','')

    documents = [Document(page_content=str(chunk)) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vector_store = Chroma.from_documents(documents,embedding=embeddings)
    query = state.get('subqueries', '')  # Avoid KeyError if 'subquery' is missing

    search_results = vector_store.similarity_search(query, k=3)

    retrieved_content = [doc.page_content for doc in search_results]

    return {'official_response': retrieved_content}


def human_escalation(state:State):
    human_eval = input('How is the response? Should it be escalated to an expert? (y/n)')
    state['human_feedback'] = human_eval
    if human_eval == 'y':
        return {'next_node':'hypo_response'}
    else:
        return {'next_node': END}
    

def decide_next_node(state:State):
    return state.get('next_node')
    




def summary(state: State):
    all_retrieved_info = state.get('official_response',"")
    query = state.get('query','')
    insert_escalation = ''
    human_escalation = state.get('human_feedback')    
    if human_escalation == 'y':
        insert_escalation = "Your previous response needs to be escalated! Modify your response to take into account that you are a lawyer who knows all " \
        "about laws within North Carolina. If you do not know the answer to a question let the user know to seek advice from a human professional"
    formatted_retrieved_content = "\n".join([doc.page_content if isinstance(doc, Document) else str(doc) for doc in all_retrieved_info])
    summary_prompt = f"""
    You are an AI assistant tasked with summarizing the following retrieved information:

    {formatted_retrieved_content}

    Provide a concise and informative summary of the content and give your references at the end in a organized and neat fashion given the query and take into account human feedback if there is any 
        Query: {query}
        Human Feedback: {insert_escalation}
    """

    response = llm.invoke(summary_prompt)

    st.write("### The following summary was generated:")
    st.write(response.content)



    return {'query_summary': response.content}
    



builder = StateGraph(State)

builder.add_node('hypo_response', hypo_response)
builder.add_node('pdf_loader', pdf_loader)
builder.add_node('split_text', split_text)
builder.add_node('build_vector_store_retrieval', build_vector_store_retrieval)
builder.add_node('human_escalation', human_escalation)
builder.add_node('summary', summary)


builder.add_edge(START, 'hypo_response')
builder.add_edge('hypo_response', 'pdf_loader')
builder.add_edge('pdf_loader', 'split_text')
builder.add_edge('split_text', 'build_vector_store_retrieval')
builder.add_edge('build_vector_store_retrieval', 'summary')
builder.add_edge('summary','human_escalation')
builder.add_conditional_edges('human_escalation', decide_next_node, ['hypo_response',END])





memory = MemorySaver()
compiled_builder = builder.compile(checkpointer=memory)



# --------- Streamlit App ------------
st.title("📚 Real Estate Assistant")
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
