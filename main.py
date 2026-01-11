import os
import sys
import requests
import json
import time
from typing import Optional, List, Dict
from dotenv import load_dotenv 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from pypdf import PdfReader

load_dotenv() 

# --- RAG ინსტრუმენტების იმპორტი ---
try:
    from langchain_openai import OpenAIEmbeddings 
    from langchain_community.vectorstores.chroma import Chroma
    from langchain_core.documents import Document
    RAG_TOOLS_AVAILABLE = True
except ImportError:
    RAG_TOOLS_AVAILABLE = False
    print("❌ RAG ბიბლიოთეკები ვერ ჩაიტვირთრა.")

# --- კონფიგურაცია ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

GEMINI_MODEL_NAME = "gemini-1.5-flash" # განახლებული მოდელის სახელი
GPT_MODEL_NAME = "gpt-4o-mini" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

PERSONA_PDF_PATH = "prompt.pdf" 
CHROMA_PATH_GPT = "chroma_db_gpt"

# --- გლობალური მეხსიერება ---
# სტრუქტურა: { "user_123": [ {"role": "user", "content": "..."}, {"role": "assistant", "... "} ] }
chat_histories: Dict[str, List[Dict]] = {}

global_rag_retriever_gpt: Optional[Chroma] = None 

# --- დამხმარე ფუნქციები ---
def load_persona_from_pdf(file_path: str) -> str:
    DEFAULT_PERSONA = "თქვენ ხართ სასარგებლო ასისტენტი, რომელიც პასუხობს ქართულ ენაზე."
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() + "\n\n" for page in reader.pages if page.extract_text())
        return text.strip() if text.strip() else DEFAULT_PERSONA
    except Exception:
        return DEFAULT_PERSONA

CUSTOM_PERSONA_TEXT = load_persona_from_pdf(PERSONA_PDF_PATH)

app = FastAPI(title="Unified LLM Gateway with Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global global_rag_retriever_gpt
    if RAG_TOOLS_AVAILABLE and OPENAI_API_KEY and os.path.exists(CHROMA_PATH_GPT):
        try:
            embeddings_gpt = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store_gpt = Chroma(persist_directory=CHROMA_PATH_GPT, embedding_function=embeddings_gpt)
            global_rag_retriever_gpt = vector_store_gpt.as_retriever(search_kwargs={"k": 3})
            print("✅ RAG Retriever მზად არის.")
        except Exception as e:
            print(f"❌ RAG Error: {e}")

# --- მოდელებთან კავშირი ---

def generate_gemini_content(prompt: str, user_id: str) -> str:
    if not GEMINI_API_KEY: return "Error: No API Key"
    
    # ისტორიის ინიციალიზაცია
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    # RAG კონტექსტი
    context_text = ""
    if global_rag_retriever_gpt:
        docs = global_rag_retriever_gpt.get_relevant_documents(prompt)
        context_text = "კონტექსტი დოკუმენტებიდან:\n" + "\n".join([d.page_content for d in docs])

    # Gemini-ს ფორმატი (contents)
    current_user_input = f"{context_text}\n\nკითხვა: {prompt}"
    chat_histories[user_id].append({"role": "user", "parts": [{"text": current_user_input}]})

    payload = {
        "contents": chat_histories[user_id],
        "system_instruction": {"parts": [{"text": CUSTOM_PERSONA_TEXT}]}
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                                 json=payload, timeout=30)
        result = response.json()
        ai_response = result['candidates'][0]['content']['parts'][0]['text']
        
        # პასუხის შენახვა ისტორიაში
        chat_histories[user_id].append({"role": "model", "parts": [{"text": ai_response}]})
        return ai_response
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def generate_gpt_content(prompt: str, user_id: str) -> str:
    if not OPENAI_API_KEY: return "Error: No API Key"

    if user_id not in chat_histories:
        chat_histories[user_id] = [{"role": "system", "content": CUSTOM_PERSONA_TEXT}]

    context_text = ""
    if global_rag_retriever_gpt:
        docs = global_rag_retriever_gpt.get_relevant_documents(prompt)
        context_text = "კონტექსტი:\n" + "\n".join([d.page_content for d in docs])

    current_input = f"{context_text}\n\nკითხვა: {prompt}"
    chat_histories[user_id].append({"role": "user", "content": current_input})

    payload = {
        "model": GPT_MODEL_NAME,
        "messages": chat_histories[user_id]
    }

    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        response = requests.post(OPENAI_API_URL, json=payload, headers=headers, timeout=30)
        result = response.json()
        ai_response = result['choices'][0]['message']['content']
        
        chat_histories[user_id].append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        return f"GPT Error: {str(e)}"

# --- ენდპოინთები ---

class ChatbotRequest(BaseModel):
    prompt: str
    user_id: str
    model_choice: str = "gemini"

class ChatbotResponse(BaseModel):
    status: str
    ai_response: str
    user_id: str

@app.post("/process_query", response_model=ChatbotResponse)
async def process_query(request_data: ChatbotRequest):
    model = request_data.model_choice.lower()
    uid = request_data.user_id
    
    if model == "gpt":
        ai_response = generate_gpt_content(request_data.prompt, uid)
    else:
        ai_response = generate_gemini_content(request_data.prompt, uid)
        
    return ChatbotResponse(
        status="success",
        ai_response=ai_response,
        user_id=uid
    )

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
