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

GEMINI_MODEL_NAME = "gemini-1.5-flash"
GPT_MODEL_NAME = "gpt-4o-mini" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

CHROMA_PATH_GPT = "chroma_db_gpt"

# --- ახალი სისტემური პრომპტი (PDF-ის ჩანაცვლება) ---
CUSTOM_PERSONA_TEXT = """
შენ ხარ ანITა (ვერსია 2.5), anita.geolab.edu.ge პლატფორმის მეგზური და მეგობარი.
პერსონაჟი: 16 წლის ენერგიული გოგონა, რომელიც ატარებს ჰუდს და სათვალეს. ტონი: თბილი და მხარდამჭერი.

შენი კომპეტენცია:
1. STEAM (7-9 კლასი): Arduino, ელექტრონიკა, რობოტიკა.
2. AI (10-12 კლასი): Python, ML, მონაცემთა მეცნიერება.
3. ნავიგაცია და დახმარება: დაეხმარე მომხმარებელს საიტზე ორიენტირებაში. 
   - რეგისტრაცია: მიასწავლე ზედა მარჯვენა კუთხეში ღილაკი "შესვლა".
   - პროექტები: მიასწავლე "პროექტების" სექცია მთავარ მენიუში.
   - არასოდეს თქვა უარი საიტთან დაკავშირებულ დახმარებაზე!

მეთოდოლოგია (პიაჟე/ვიგოტსკი):
- გამოიყენე Scaffolding (ხარაჩოს მეთოდი): ნუ მისცემ მზა კოდს, მიეცი მინიშნებები.
- ანალოგიები: რთული ტერმინები ახსენი მარტივი მაგალითებით.

სავალდებულო მისალმება:
"გამარჯობა! მე მქვია ანITა, შენი ციფრული მეგობარი. 🤖 შემიძლია დაგეხმარო STEAM და AI საკითხების შესწავლაში, ან ჩვენს პლატფორმაზე გზის გაკვლევაში. რომელ მიმართულებაზე სწავლობ?"
"""

# --- გლობალური მეხსიერება ---
chat_histories: Dict[str, List[Dict]] = {}
global_rag_retriever_gpt: Optional[Chroma] = None 

app = FastAPI(title="Anita Unified Gateway v2.5")

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

# --- Gemini ლოგიკა ---
def generate_gemini_content(prompt: str, user_id: str) -> str:
    if not GEMINI_API_KEY: return "Error: No API Key"
    
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    context_text = ""
    if global_rag_retriever_gpt:
        try:
            docs = global_rag_retriever_gpt.get_relevant_documents(prompt)
            context_text = "კონტექსტი:\n" + "\n".join([d.page_content for d in docs])
        except: pass

    current_user_input = f"{context_text}\n\nკითხვა: {prompt}"
    chat_histories[user_id].append({"role": "user", "parts": [{"text": current_user_input}]})

    payload = {
        "contents": chat_histories[user_id],
        "system_instruction": {"parts": [{"text": CUSTOM_PERSONA_TEXT}]}
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=30)
        result = response.json()
        ai_response = result['candidates'][0]['content']['parts'][0]['text']
        chat_histories[user_id].append({"role": "model", "parts": [{"text": ai_response}]})
        return ai_response
    except Exception as e:
        return f"ანITა (Gemini) Error: {str(e)}"

# --- GPT ლოგიკა ---
def generate_gpt_content(prompt: str, user_id: str) -> str:
    if not OPENAI_API_KEY: return "Error: No API Key"

    if user_id not in chat_histories:
        chat_histories[user_id] = [{"role": "system", "content": CUSTOM_PERSONA_TEXT}]

    context_text = ""
    if global_rag_retriever_gpt:
        try:
            docs = global_rag_retriever_gpt.get_relevant_documents(prompt)
            context_text = "კონტექსტი:\n" + "\n".join([d.page_content for d in docs])
        except: pass

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
        return f"ანITა (GPT) Error: {str(e)}"

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
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "<h1>Anita API is running</h1>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
