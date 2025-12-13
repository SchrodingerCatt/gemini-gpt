# ingest_gpt.py (დარწმუნდით, რომ ეს ფაილი გამოიყენება)

import os
from dotenv import load_dotenv 

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma

load_dotenv() 

# --- კონფიგურაცია ---
DATA_DIR = "Steam" 
CHROMA_PATH = "chroma_db_gpt" #  ყველაზე მნიშვნელოვანი ცვლილება აქ!

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY ვერ იქნა ნაპოვნი.")
    print("გთხოვთ, შექმნათ .env ფაილი და ჩაწეროთ მასში: OPENAI_API_KEY='თქვენი გასაღები'")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def ingest_data():
    """
    კითხულობს PDF-ებს, ყოფს ტექსტს (chunking), ვექტორიზაციას უკეთებს და ინახავს ChromaDB-ში.
    """
    if not OPENAI_API_KEY:
          print("❌ ინდექსირება გაუქმებულია: OpenAI API გასაღები აკლია.")
          return
          
    print(f"--- 1. დაწყებულია მონაცემების წაკითხვა საქაღალდიდან: {DATA_DIR} ---")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ შეცდომა: საქაღალდე '{DATA_DIR}' ვერ მოიძებნა.")
        return

    try:
        loader = PyPDFDirectoryLoader(DATA_DIR)
        documents = loader.load()
    except Exception as e:
        print(f"❌ შეცდომა დოკუმენტების წაკითხვისას: {e}")
        return

    if not documents:
        print(f"⚠️ არცერთი დოკუმენტი არ მოიძებნა ინდექსირებისთვის საქაღალდეში: {DATA_DIR}.")
        return
        
    print(f"✅ მოიძებნა {len(documents)} დოკუმენტი წასაკითხად.")

    print("--- 2. მიმდინარეობს ტექსტის ფრაგმენტებად დაყოფა (Chunking) ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"✅ შექმნილია {len(texts)} ტექსტური ფრაგმენტი (chunk).")

    print("--- 3. მიმდინარეობს ვექტორიზაცია (Embedding) და შენახვა ChromaDB-ში ---")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # ეს ხაზი ქმნის და ინახავს ChromaDB-ს CHROMA_PATH-ში ("chroma_db_gpt")
    Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ ინდექსირება წარმატებით დასრულდა. ვექტორული ბაზა შენახულია: {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_data()