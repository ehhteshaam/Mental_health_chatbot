from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import gradio as gr
import os

# --- Setup LLM ---
def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],  # Store this in Hugging Face Secrets
        model_name="llama-3-70b-8192"
    )

# --- Create or Load Vector DB ---
def create_vector_db(pdf_path, db_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    return vector_db

# Paths
pdf_file = "mental_health_document.pdf"
db_path = "chroma_db"

# Init
llm = initialize_llm()

if not os.path.exists(db_path):
    vector_db = create_vector_db(pdf_file, db_path)
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Setup Retriever + Prompt
retriever = vector_db.as_retriever()
prompt_template = """You are a compassionate mental health chatbot. Answer thoughtfully:
{context}
User: {question}
Chatbot:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Chat Function
def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input", history
    response = qa_chain.run(user_input)
    history.append((user_input, response))
    return response

# --- Gradio UI ---
with gr.Blocks(theme='gradio/soft') as app:
    gr.Markdown("# 🧠 Mental Health Chatbot 🤖")
    gr.Markdown("### A supportive AI space. Please note: Not a substitute for professional care.")
    chatbot = gr.ChatInterface(fn=chatbot_response, title="Mental Health Chatbot")
    gr.Markdown("⚠️ *If you're facing serious emotional issues, please consult a licensed mental health professional.*")

app.launch()
