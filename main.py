import os
import sys
import requests
import json
import time
from typing import Optional
# from dotenv import load_dotenv # <--- წაშლილია!

# --- FastAPI და HTML იმპორტები ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from pypdf import PdfReader

# --- RAG ინსტრუმენტების იმპორტი ---
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_openai import OpenAIEmbeddings 
    from langchain_community.vectorstores.chroma import Chroma
    from langchain_core.documents import Document
    RAG_TOOLS_AVAILABLE = True
except ImportError:
    RAG_TOOLS_AVAILABLE = False
    print("❌ RAG ბიბლიოთეკები ვერ ჩაიტვირთა. RAG არააქტიურია.")
    
# --- კონფიგურაცია: გასაღებების მოტანა გარემოს ცვლადებიდან ---
# ამ ეტაპზე ვეყრდნობით მხოლოდ ჰოსტინგ გარემოს ცვლადებს (Render)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
# (აქ არაფერი ეწერება load_dotenv-თან დაკავშირებით)

# --- მოდელების და RAG-ის პარამეტრები ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GPT_MODEL_NAME = "gpt-4o-mini" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# --- RAG-ის და პერსონის კონფიგურაცია ---
PERSONA_PDF_PATH = "prompt.pdf" 
CHROMA_PATH_GPT = "chroma_db_gpt"

# გლობალური ობიექტები
global_rag_retriever_gemini: Optional[Chroma.as_retriever] = None
global_rag_retriever_gpt: Optional[Chroma.as_retriever] = None 

# --- ფუნქცია პერსონის PDF-დან ჩასატვირთად ---
def load_persona_from_pdf(file_path: str) -> str:
    """კითხულობს მთელ ტექსტს PDF ფაილიდან pypdf-ის გამოყენებით."""
    DEFAULT_PERSONA = "თქვენ ხართ სასარგებლო ასისტენტი, რომელიც პასუხობს ქართულ ენაზე."
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() + "\n\n" for page in reader.pages if page.extract_text())
        if not text.strip():
            print(f"❌ ERROR: PDF ფაილი '{file_path}' ცარიელია. გამოყენებულია დეფოლტური პერსონა.")
            return DEFAULT_PERSONA
        print(f"✅ პერსონის ტექსტი წარმატებით ჩაიტვირთა {file_path}-დან.")
        return text.strip()
    except Exception as e:
        print(f"❌ ERROR: პერსონის PDF-ის წაკითხვისას შეცდომა: {e}. გამოყენებულია დეფოლტური პერსონა.")
        return DEFAULT_PERSONA


CUSTOM_PERSONA_TEXT = load_persona_from_pdf(PERSONA_PDF_PATH)

# --- FastAPI აპლიკაციის ინიციალიზაცია ---
app = FastAPI(title="Unified LLM Gateway (Gemini & GPT)", version="2.0")

# --- Startup ლოგიკა: RAG ინიციალიზაცია ---
@app.on_event("startup")
async def startup_event():
    global global_rag_retriever_gemini
    global global_rag_retriever_gpt
    
    if not RAG_TOOLS_AVAILABLE:
        print("RAG ინიციალიზაცია გამოტოვებულია.")
        return
        
    # 2. 🤖 GPT RAG ინიციალიზაცია 
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        if os.path.exists(CHROMA_PATH_GPT):
            try:
                embeddings_gpt = OpenAIEmbeddings(model="text-embedding-3-small")
                vector_store_gpt = Chroma(
                    persist_directory=CHROMA_PATH_GPT, 
                    embedding_function=embeddings_gpt
                )
                global_rag_retriever_gpt = vector_store_gpt.as_retriever(search_kwargs={"k": 3})
                print(f"✅ GPT RAG Retriever წარმატებით ჩაიტვირთა: {CHROMA_PATH_GPT}")
            except Exception as e:
                print(f"❌ ERROR: GPT ChromaDB-ის ჩატვირთვა ვერ მოხერხდა: {e}.")
        else:
            print(f"⚠️ WARNING: ვექტორული ბაზა {CHROMA_PATH_GPT} ვერ მოიძებნა. GPT RAG არააქტიურია. გაუშვით ingest_gpt.py")
    else:
        print("❌ ERROR: OpenAI API გასაღები ვერ მოიძებნა.")


    # 1. 💎 Gemini RAG ინიციალიზაცია (იყენებს GPT-ის წარმატებულ ბაზას)
    if GEMINI_API_KEY and OPENAI_API_KEY:
        if global_rag_retriever_gpt:
            global_rag_retriever_gemini = global_rag_retriever_gpt
            print(f"✅ Gemini RAG Retriever წარმატებით დაყენდა GPT-ის ბაზაზე: {CHROMA_PATH_GPT}")
        elif os.path.exists(CHROMA_PATH_GPT): 
            try:
                embeddings_gpt = OpenAIEmbeddings(model="text-embedding-3-small") 
                vector_store = Chroma(
                    persist_directory=CHROMA_PATH_GPT, 
                    embedding_function=embeddings_gpt
                )
                global_rag_retriever_gemini = vector_store.as_retriever(search_kwargs={"k": 3})
            except Exception as e:
                print(f"❌ ERROR: Gemini RAG ვერ ჩაიტვირთა GPT-ის ბაზიდან: {e}. არააქტიურია.")
                global_rag_retriever_gemini = None 
        else:
            print(f"⚠️ WARNING: ვექტორული ბაზა {CHROMA_PATH_GPT} ვერ მოიძებნა. Gemini RAG არააქტიურია.")
    else:
        print("❌ ERROR: Gemini ან OpenAI API გასაღები ვერ მოიძებნა.")


# --- CORS Middleware დამატება ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8080"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HTML ფაილის ჩატვირთვა და სერვირება ---
try:
    with open("index.html", "r", encoding="utf-8") as f:
        HTML_CONTENT = f.read()
except FileNotFoundError:
    HTML_CONTENT = "<h1>FastAPI API მუშაობს, მაგრამ ფრონტენდის (index.html) ფაილი ვერ მოიძებნა.</h1>"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

# --- მონაცემთა მოდელები ---
class ChatbotRequest(BaseModel):
    prompt: str
    user_id: str
    model_choice: str = "gemini"

class ChatbotResponse(BaseModel):
    status: str
    processed_prompt: str
    ai_response: str
    result_data: dict

# --- 1. Gemini API-ს გამოძახება (RAG ლოგიკით) ---
def generate_gemini_content(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "ERROR: Gemini API გასაღები ვერ მოიძებნა."
        
    rag_context = ""
    is_rag_active = global_rag_retriever_gemini is not None
    
    if is_rag_active:
        try:
            docs: list[Document] = global_rag_retriever_gemini.get_relevant_documents(prompt)
            context_text = "\n---\n".join([doc.page_content for doc in docs])
            rag_context = (
                "თქვენ მოგეცემათ დამატებითი კონტექსტი 'DOCUMENTS'-ის სექციაში. "
                "გამოიყენეთ ეს ინფორმაცია, რომ უპასუხოთ შეკითხვას.\n\n"
                f"--- DOCUMENTS ---\n{context_text}\n---"
            )
        except Exception:
            rag_context = "RAG retrieval failed."

    final_prompt = f"{rag_context}\n\nმომხმარებლის შეკითხვა: {prompt}"

    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "role": "user",  
                "parts": [{"text": f"შემდეგი ტექსტი განსაზღვრავს თქვენს მთავარ პერსონას. მკაცრად მიჰყევით მას:\n\n---\n{CUSTOM_PERSONA_TEXT}\n---"}]
            },
            {
                "role": "user",
                "parts": [{"text": final_prompt}]
            }
        ]
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                headers=headers, 
                data=json.dumps(payload),
                timeout=30 
            )
            
            if response.status_code >= 400:
                error_msg = f"Gemini API-მ დააბრუნა {response.status_code} შეცდომა."
                try:
                    error_detail = response.json()
                    error_msg += f" დეტალები: {error_detail.get('error', {}).get('message', 'დეტალური შეტყობინება ვერ მიიღეს.')}"
                except json.JSONDecodeError:
                    pass
                return f"ERROR: {error_msg}"

            response.raise_for_status() 
            result = response.json()
            
            candidate = result.get('candidates', [{}])[0]
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                return candidate['content']['parts'][0]['text']
            
            return f"Gemini API-მ დააბრუნა არასტანდარტული პასუხი."

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return f"ERROR: Gemini API-სთან დაკავშირება ვერ მოხერხდა. შეცდომა: {e}"
        except Exception as e:
            return f"ERROR: მოულოდნელი შეცდომა: {e}"
            
    return "ERROR: პასუხი ვერ იქნა გენერირებული."

# --- 2. GPT API-ს გამოძახება (RAG ლოგიკით) ---
def generate_gpt_content(prompt: str) -> str:
    if not OPENAI_API_KEY:
        return "ERROR: GPT API გასაღები ვერ მოიძებნა."
    
    rag_context = ""
    is_rag_active = global_rag_retriever_gpt is not None
    
    if is_rag_active:
        try:
            docs: list[Document] = global_rag_retriever_gpt.get_relevant_documents(prompt)
            context_text = "\n---\n".join([doc.page_content for doc in docs])
            
            rag_context = (
                f"გამოიყენეთ შემდეგი კონტექსტი პასუხის გასაცემად. თუ პასუხი მოცემულ კონტექსტში არ არის, "
                f"მაშინ უპასუხეთ ზოგადი ცოდნის საფუძველზე: \n\n--- DOCUMENTS ---\n{context_text}\n---"
            )
        except Exception as e:
            print(f"❌ ERROR: GPT RAG Retrieval-ის შეცდომა: {e}")
            rag_context = "RAG retrieval failed."

    final_user_prompt = f"{rag_context}\n\nმომხმარებლის შეკითხვა: {prompt}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": GPT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": f"{CUSTOM_PERSONA_TEXT}"},
            {"role": "user", "content": final_user_prompt}
        ]
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENAI_API_URL, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=30 
            )
            
            if response.status_code >= 400:
                error_msg = f"OpenAI API-მ დააბრუნა {response.status_code} შეცდომა."
                try:
                    error_detail = response.json()
                    error_msg += f" დეტალები: {error_detail.get('error', {}).get('message', 'დეტალური შეტყობინება ვერ მიიღეს.')}"
                except json.JSONDecodeError:
                    pass
                return f"ERROR: {error_msg}"

            response.raise_for_status() 
            result = response.json()
            
            if result.get('choices'):
                return result['choices'][0]['message']['content']
            
            return f"OpenAI API-მ დააბრუნა არასტანდარტული პასუხი."

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return f"ERROR: OpenAI API-სთან დაკავშირება ვერ მოხერხდა. შეცდომა: {e}"
        except Exception as e:
            return f"ERROR: მოულოდნელი შეცდომა: {e}"
            
    return "ERROR: პასუხი ვერ იქნა გენერირებული."


# --- API ენდპოინტები ---
@app.get("/status")
def read_root():
    rag_gemini_status = "აქტიურია" if global_rag_retriever_gemini else "არააქტიურია (ბაზა ვერ მოიძებნა)"
    rag_gpt_status = "აქტიურია" if global_rag_retriever_gpt else "არააქტიურია (გაუშვით ingest_gpt.py)"
    
    return {
        "message": "API მუშაობს!", 
        "Gemini_Model": GEMINI_MODEL_NAME,
        "GPT_Model": GPT_MODEL_NAME,
        "RAG_Status_Gemini": rag_gemini_status,
        "RAG_Status_GPT": rag_gpt_status,
        "Note": "გამოიყენეთ /process_query ენდფოინთი, model_choice: 'gemini' ან 'gpt'"
    }

@app.post("/process_query", response_model=ChatbotResponse)
async def process_query(
    request_data: ChatbotRequest
):
    model_choice = request_data.model_choice.lower()
    
    if model_choice == "gemini":
        ai_response = generate_gemini_content(request_data.prompt)
        used_model_name = GEMINI_MODEL_NAME
        is_rag_active = global_rag_retriever_gemini is not None
    elif model_choice == "gpt":
        ai_response = generate_gpt_content(request_data.prompt)
        used_model_name = GPT_MODEL_NAME
        is_rag_active = global_rag_retriever_gpt is not None
    else:
        ai_response = generate_gemini_content(request_data.prompt)
        used_model_name = GEMINI_MODEL_NAME
        is_rag_active = global_rag_retriever_gemini is not None
        
    response_data = {
        "user": request_data.user_id,
        "length": len(request_data.prompt),
        "is_rag_active": is_rag_active,
        "used_model": used_model_name
    }
    
    return ChatbotResponse(
        status="success",
        processed_prompt=f"თქვენი მოთხოვნა დამუშავებულია {used_model_name}-ის მიერ.",
        ai_response=ai_response,
        result_data=response_data,
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8090)))
