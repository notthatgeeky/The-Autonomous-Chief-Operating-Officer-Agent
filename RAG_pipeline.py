import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

REPORTS_DIR = "reports"
CHROMA_PATH = "chroma_db"

def load_documents():
    print(f"Scanning '{REPORTS_DIR}' for PDF reports...")
    documents = []
    
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        print(f"Created directory '{REPORTS_DIR}'. Please put some dummy PDFs inside and run again.")
        return documents

    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith('.pdf'):
            filepath = os.path.join(REPORTS_DIR, filename)
            print(f"Loading: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            
    print(f"Loaded {len(documents)} pages in total.")
    return documents

def split_text(documents):
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split documents into {len(chunks)} text chunks.")
    return chunks

def save_to_chroma(chunks):
    print("Initializing embedding model and saving to ChromaDB...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Successfully saved {len(chunks)} embeddings to '{CHROMA_PATH}' database.")
    return db

def test_query(query):
    print(f"\n--- Testing Query: '{query}' ---")
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
    results = db.similarity_search(query, k=3)
    
    if not results:
        print("No results found.")
    else:
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Source: {result.metadata.get('source', 'Unknown')}):")
            print(f"\"{result.page_content[:200]}...\"") # Print first 200 chars

if __name__ == "__main__":
    print("--- Starting RAG Ingestion Pipeline ---")
    
    docs = load_documents()
    
    if len(docs) > 0:
        chunks = split_text(docs)
        save_to_chroma(chunks)
        
        test_query("What are the main market trends?")
        
    print("\n--- Pipeline Execution Finished ---")