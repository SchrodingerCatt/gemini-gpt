import os
import sys
import shutil

# 1. SQLite ფიქსაცია სერვერისთვის (აუცილებელია Chroma-სთვის)
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

import google.generativeai as genai
from dotenv import load_dotenv 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma 
from langchain_core.embeddings import Embeddings
from typing import List

# კონფიგურაცია 
load_dotenv() 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# აბსოლუტური გზების განსაზღვრა
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "Steam") 
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db") 

genai.configure(api_key=GEMINI_API_KEY)


#  ფუნქცია ხელმისაწვდომი ემბედინგ მოდელის საპოვნელად 
def get_available_embedding_model():
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            return m.name
    return None

AVAILABLE_MODEL = get_available_embedding_model()
if AVAILABLE_MODEL:
    print(f" ნაპოვნია ხელმისაწვდომი მოდელი: {AVAILABLE_MODEL}")
else:
    print(" შეცდომა: ემბედინგის მოდელი ვერ მოიძებნა!")

#  ემბედინგების კლასი 
class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name=AVAILABLE_MODEL):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 50 # შევამციროთ ზომა სტაბილურობისთვის
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                # ზოგჯერ v1beta-ში სჭირდება task_type-ის გარეშე ან სხვადასხვა ვერსიით
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                all_embeddings.extend(result['embedding'])
            except Exception as e:
                print(f" შეცდომა ბატჩზე {i}: {e}")
                raise e
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

def ingest_documents():
    if not AVAILABLE_MODEL:
        return

    if not os.path.exists(DOCS_DIR):
        print(f" საქაღალდე '{DOCS_DIR}' არ არსებობს.")
        return

    # PDF-ების ჩატვირთვა
    documents = []
    print(f" PDF-ების დამუშავება...")
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    
    for filename in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(DOCS_DIR, filename))
            documents.extend(loader.load())
            print(f"    done {filename}")
        except Exception as e:
            print(f"    fail {filename}: {e}")

    # დაჭრა
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f" დაიყო {len(chunks)} ნაწილად.")

    # ბაზის შექმნა
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=GeminiEmbeddings(),
            persist_directory=CHROMA_PATH
        )
        print(f" ბაზა წარმატებით შეიქმნა მოდელით: {AVAILABLE_MODEL}")
    except Exception as e:
        print(f" კრიტიკული შეცდომა ბაზის შექმნისას: {e}")

if __name__ == "__main__":
    ingest_documents()
