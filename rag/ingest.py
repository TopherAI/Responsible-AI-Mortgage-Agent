import os
import hashlib
import zipfile
import requests
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Config ─────────────────────────────────────────────────
GUIDELINES_DIR  = Path("agents/data/guidelines")
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "mortgage_compliance"

AUTO_DOWNLOAD = {}  # all files present in repo, no download needed

GUIDELINE_FILES = {
    "fannie_mae_selling_guide.pdf":      "Fannie Mae Selling Guide",
    "freddie_mac_guide_1of2.pdf":        "Freddie Mac Seller/Servicer Guide Part 1",
    "freddie_mac_guide_2of2.pdf":        "Freddie Mac Seller/Servicer Guide Part 2",
    "fha_handbook_4000_1.pdf":           "FHA Single Family Handbook 4000.1",
    "va_lenders_handbook_26_7.pdf":      "VA Lenders Handbook 26-7",
}

# ── Step 0a: Auto-download large files ────────────────────
def download_large_files():
    print("\n── Checking for downloads ──")
    for filename, url in AUTO_DOWNLOAD.items():
        dest = GUIDELINES_DIR / filename
        if dest.exists():
            print(f"  ✓ Already exists: {filename}")
            continue
        print(f"  → Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  ✓ Downloaded: {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Failed: {filename}: {e}")

# ── Step 0b: Unzip compressed files ───────────────────────
def unzip_guidelines():
    print("\n── Checking for compressed files ──")
    for zip_path in GUIDELINES_DIR.glob("*.zip"):
        print(f"  → Unzipping {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(GUIDELINES_DIR)
        print(f"  ✓ Extracted {zip_path.name}")

# ── Step 1: Load PDFs ──────────────────────────────────────
def load_documents():
    all_docs = []
    print("\n── Loading PDFs ──")
    for filename, source_label in GUIDELINE_FILES.items():
        filepath = GUIDELINES_DIR / filename
        if not filepath.exists():
            print(f"  ⚠ MISSING: {filename} — skipping")
            continue
        print(f"  ✓ Loading: {source_label}")
        loader = PyPDFLoader(str(filepath))
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source_label"] = source_label
            doc.metadata["filename"]     = filename
            doc.metadata["doc_id"]       = hashlib.md5(
                source_label.encode()
            ).hexdigest()[:8]
        all_docs.extend(docs)
        print(f"    → {len(docs)} pages loaded")
    return all_docs

# ── Step 2: Chunk ──────────────────────────────────────────
def chunk_documents(docs):
    print("\n── Chunking Documents ──")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  → {len(chunks)} total chunks created")
    return chunks

# ── Step 3: Embed + Store ──────────────────────────────────
def embed_and_store(chunks):
    print("\n── Embedding & Storing in ChromaDB ──")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.persist()
    print(f"  ✓ {len(chunks)} chunks embedded and saved to {CHROMA_DIR}/")
    return vectorstore

# ── Step 4: Smoke Test ─────────────────────────────────────
def smoke_test(vectorstore):
    print("\n── Smoke Test Queries ──")
    test_queries = [
        "What is the maximum debt-to-income ratio allowed?",
        "What is the minimum credit score for an FHA loan?",
        "What are VA loan occupancy requirements?",
        "What are Freddie Mac income documentation requirements?",
    ]
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\n  Q: {query}")
        for r in results:
            print(
                f"    [{r.metadata.get('source_label','?')} "
                f"p.{r.metadata.get('page','?')}] "
                f"{r.page_content[:120].strip()}..."
            )

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 55)
    print("  Compliance RAG Ingestion Pipeline")
    print("  Responsible AI Mortgage Agent")
    print("═" * 55)

    GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    download_large_files()
    unzip_guidelines()

    docs = load_documents()
    if not docs:
        print("\n✗ No documents loaded. Check agents/data/guidelines/ and retry.")
        exit(1)

    chunks      = chunk_documents(docs)
    vectorstore = embed_and_store(chunks)
    smoke_test(vectorstore)

    print("\n✓ Ingestion complete. ChromaDB ready for LangGraph queries.")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Location   : {CHROMA_DIR}/")
    print(f"  Chunks     : {len(chunks)}")
