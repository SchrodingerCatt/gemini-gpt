import sys
import os

# 1. SQLite ფიქსაცია (აუცილებელია სერვერისთვის)
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

import datetime
import uvicorn
import base64
import google.generativeai as genai
from openai import OpenAI
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pypdf import PdfReader
from langchain_chroma import Chroma
from dotenv import load_dotenv
from ingest_gemini import GeminiEmbeddings 

load_dotenv()
app = FastAPI()

# 2. CORS კონფიგურაცია - მუშაობს ყველგან (localhost, render, vercel და ა.შ.)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=False, # აუცილებელია False, როცა allow_origins=["*"]
    allow_methods=["*"], 
    allow_headers=["*"]
)

# 3. კონფიგურაცია აბსოლუტური გზებით
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_new_v25")
PERSONA_PDF = os.path.join(BASE_DIR, "prompt.pdf")
MY_GEMINI_MODEL = "gemini-2.5-flash" # შენი მოდელი

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
client_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ლოკალზე ტესტირებისთვის: index.html-ის პირდაპირ გახსნა
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"შეცდომა ფაილის წაკითხვისას: {str(e)}"

def load_persona():
    base_text = ""
    if os.path.exists(PERSONA_PDF):
        try:
            reader = PdfReader(PERSONA_PDF)
            base_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except: base_text = "შენ ხარ ანITა."
    
    return f"""
    {base_text}
    
    მკაცრი წესები (Strict Handling):
    1. იყავი მაქსიმალურად მოკლე. თუ მომხმარებელი რამეს გატყობინებს (მაგ: 'მე ვსწავლობ პითონს'), უბრალოდ დაუდასტურე მეგობრულად და ნუ დაიწყებ ახსნას ან დავალებების მოცემას.
    2. არ ახსნა არცერთი ტექნიკური ტერმინი (მაგ: append, ცვლადი), თუ მომხმარებელმა პირდაპირ არ გკითხა: "ამიხსენი რა არის X".
    3. თუ გკითხეს "რა მქვია?", უპასუხე მხოლოდ სახელით.
    4. ნუ დაასრულებ პასუხს კითხვით (მაგ: "გინდა დავიწყოთ?"), თუ მომხმარებელს დახმარება არ უთხოვია.
    5. თუ კითხვა საერთოდ არ ეხება შენს სფეროს, გამოიყენე სტანდარტული უარი.
    6.გამოიყენე ემოჯები 😊, 🤖, ✨, რომ საუბარი უფრო ბუნებრივი იყოს.
    """

SYSTEM_INSTRUCTION = load_persona()
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=GeminiEmbeddings())
db = {} 

def get_user_data(user_id: str):
    today = str(datetime.date.today())
    if user_id not in db or db[user_id]["date"] != today:
        db[user_id] = {"history": [], "media_count": 0, "date": today}
    return db[user_id]

@app.post("/process_query")
async def chat_endpoint(
    user_id: str = Form(...),
    prompt: str = Form(...),
    model_choice: str = Form("gemini"),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    user_info = get_user_data(user_id)
    # --- დაამატე ეს ხაზები აქედან ---
    if image:
        print(f"[DEBUG] მოვიდა ფოტო: {image.filename}", flush=True)
    if audio:
        print(f"[DEBUG] მოვიდა აუდიო: {audio.filename}", flush=True)
    if not image and not audio:
        print("[DEBUG] მედია ფაილი არ მოსულა (მხოლოდ ტექსტი)", flush=True)
    # --- აქამდე ---
    has_media = image is not None or audio is not None
    if has_media and user_info["media_count"] >= 1000:
        raise HTTPException(status_code=429, detail="მედია ლიმიტი ამოწურულია.")

    # --- Chroma DB-ში ძებნა და დებაგინგი ---
    docs = vector_store.similarity_search(prompt, k=2)
    
    print(f"\n[DEBUG] Chroma DB-მ იპოვა {len(docs)} შესაბამისი ნაწყვეტი.")
    for i, d in enumerate(docs):
        print(f"ნაწყვეტი {i+1}: {d.page_content[:150]}...")
    # --------------------------------------

    context = "\n".join([d.page_content for d in docs])
    full_query = f"დამხმარე მასალა: {context}\n\nმომხმარებელი: {prompt}"

    try:
        if model_choice == "gpt":
            messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
            content = [{"type": "text", "text": full_query}]
            if image:
                img_b64 = base64.b64encode(await image.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            messages.append({"role": "user", "content": content})
            res = client_openai.chat.completions.create(model="gpt-4o", messages=messages)
            ai_text = res.choices[0].message.content
        else:
            model = genai.GenerativeModel(model_name=MY_GEMINI_MODEL, system_instruction=SYSTEM_INSTRUCTION)
            chat_session = model.start_chat(history=user_info["history"])
            parts = [full_query]
            if image: parts.append({"mime_type": image.content_type, "data": await image.read()})
            if audio: parts.append({"mime_type": audio.content_type, "data": await audio.read()})
            response = chat_session.send_message(parts)
            ai_text = response.text

        user_info["history"].append({"role": "user", "parts": [prompt]})
        user_info["history"].append({"role": "model", "parts": [ai_text]})
        if has_media: user_info["media_count"] += 1
        return {"response": ai_text, "media_remaining": 1000 - user_info["media_count"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    uvicorn.run(app, host="0.0.0.0", port=port)
