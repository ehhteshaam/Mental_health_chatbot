from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import gradio as gr
import os

# --- New Imports for Voice ---
import speech_recognition as sr
from gtts import gTTS
import tempfile

# --- Setup LLM ---
def initialize_llm():
    # Ensure your API key is set in your environment variables
    # or replace os.environ["GROQ_API_KEY"] with your actual key string for testing
    return ChatGroq(
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"), 
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

# DB Initialization Check
if not os.path.exists(db_path):
    # Only create if PDF exists, otherwise handle gracefully for the demo
    if os.path.exists(pdf_file):
        vector_db = create_vector_db(pdf_file, db_path)
    else:
        # Fallback for when PDF isn't present (e.g. first run without file)
        print(f"Warning: {pdf_file} not found. Vector DB creation skipped.")
        vector_db = None 
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Setup Retriever + Prompt
if vector_db:
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health chatbot. Answer thoughtfully:
    {context}
    User: {question}
    Chatbot:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
else:
    qa_chain = None

# --- Voice Helper Functions ---

def transcribe_audio(audio_path):
    """Converts recorded audio to text using Google Speech Recognition."""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Sorry, there was an issue connecting to the speech service."
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def text_to_speech(text):
    """Converts text to an MP3 file using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        # Create a temp file to save the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# --- Chat Core Logic ---

def process_interaction(user_text, user_audio, history):
    """Handles text or audio input, generates response, and creates audio output."""
    
    # 1. Handle Input (Audio takes priority if provided)
    if user_audio is not None:
        user_input = transcribe_audio(user_audio)
    else:
        user_input = user_text

    if not user_input or user_input.strip() == "":
        return history, None, ""

    # 2. Generate Text Response
    greetings = ["hi", "hello", "hey", "how are you", "what's up", "good morning", "good evening"]
    
    if user_input.lower().strip() in greetings:
        reply = "Hello! I'm here to support you. How are you feeling today?"
    elif qa_chain:
        try:
            reply = qa_chain.run(user_input)
        except Exception as e:
            reply = "I'm having trouble accessing my knowledge base right now."
    else:
        reply = "My knowledge base isn't ready. Please ensure the PDF is loaded."

    # 3. Generate Audio Response
    audio_response_path = text_to_speech(reply)

    # 4. Update History
    history.append((user_input, reply))

    # Return: History update, Audio path, Clear text input, Clear audio input
    return history, audio_response_path, "", None


# --- Gradio UI (Custom Blocks) ---
with gr.Blocks(theme='gradio/soft') as app:
    gr.Markdown("# 🧠 Mental Health Chatbot (Voice Enabled) 🤖")
    gr.Markdown("### Speak or Type. I will listen and answer back.")

    # Chat Interface
    chatbot_display = gr.Chatbot(label="Conversation")
    
    # Audio Output (Auto-plays the AI response)
    ai_audio_output = gr.Audio(label="AI Voice", type="filepath", autoplay=True)

    with gr.Row():
        # Text Input
        msg_input = gr.Textbox(
            show_label=False, 
            placeholder="Type your message here...", 
            scale=4
        )
        submit_btn = gr.Button("Send", scale=1)

    # Audio Input
    mic_input = gr.Audio(
        source="microphone", 
        type="filepath", 
        label="🎤 Speak to Chatbot"
    )

    # State for history
    chat_history = gr.State([])

    # Event Handlers
    
    # 1. Text Submission
    submit_btn.click(
        fn=process_interaction,
        inputs=[msg_input, mic_input, chat_history],
        outputs=[chatbot_display, ai_audio_output, msg_input, mic_input]
    )
    
    # 2. Text "Enter" Key Submission
    msg_input.submit(
        fn=process_interaction,
        inputs=[msg_input, mic_input, chat_history],
        outputs=[chatbot_display, ai_audio_output, msg_input, mic_input]
    )

    # 3. Audio Submission (Triggered when recording stops/uploads)
    # Note: Gradio's Audio component behavior varies. 
    # Usually users prefer to record and then click a button to send, 
    # but we can also wire it to a specific 'Transcribe' button if preferred.
    # Here, we use the "Send" button to process whatever is in the inputs.

    gr.Markdown("⚠️ *If you're facing serious emotional issues, please consult a licensed mental health professional.*")

app.launch()
