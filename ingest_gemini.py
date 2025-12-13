# ingest_gemini.py (განახლებული ვერსია)

import os
import sys
from dotenv import load_dotenv 

# 💥 დაემატა GoogleGenAI, თუ LangChain-ის ვერსია მოძველებულია
try:
    from google import genai as GoogleGenAI
except ImportError:
    pass 
    
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma 

load_dotenv() 

# --- კონფიგურაცია ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DOCS_DIR = "Steam" 
CHROMA_PATH = "chroma_db" 

if not GEMINI_API_KEY:
    print("❌ ERROR: Gemini API გასაღები ვერ მოიძებნა. ინდექსირება შეუძლებელია.")
    sys.exit(1)

# 💥 იძულებით დაყენება LangChain-ისთვის და GenAI-სთვის
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY 
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY 


def ingest_documents():
    """კითხულობს PDF-ებს, ანაწილებს მათ და ინახავს ChromaDB-ში."""
    
    # ... (დოკუმენტების ჩატვირთვის ლოგიკა უცვლელია) ...
    
    if not os.path.exists(DOCS_DIR):
        print(f"❌ ERROR: დოკუმენტების საქაღალდე '{DOCS_DIR}' ვერ მოიძებნა.")
        return

    documents = []
    print(f"🔄 დოკუმენტების ჩატვირთვა საქაღალდიდან: {DOCS_DIR}...")
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    
    for filename in pdf_files:
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"    ✅ ჩაიტვირთა: {filename}")
        except Exception as e:
            print(f"    ❌ შეცდომა ჩატვირთვისას {filename}: {e}")
            
    if not documents:
        print("❌ ERROR: ვერ მოიძებნა PDF ფაილები ჩასატვირთად.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📊 დოკუმენტები დანაწილდა {len(chunks)} ფრაგმენტად (Chunks).")
    
    print("💾 ვექტორების გენერაცია და ChromaDB-ში შენახვა...")
    try:
        # 💥 მნიშვნელოვანი ცვლილება: API გასაღების ექსპლიციტური გადაცემა
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            api_key=GEMINI_API_KEY 
        ) 
        
        vector_store = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=CHROMA_PATH
        )
        vector_store.persist()
        print(f"✅ ინდექსირება დასრულდა! მონაცემთა ბაზა შენახულია: {CHROMA_PATH}")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: ვექტორების შექმნა ვერ მოხერხდა. დეტალები: {e}")
        sys.exit(1)


if __name__ == "__main__":
    ingest_documents()