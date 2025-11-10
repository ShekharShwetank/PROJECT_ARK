import os
import time
import argparse
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- CONFIGURATION ---
# We no longer hardcode paths here, they will come from command-line arguments.
DB_PATH = "db"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

def main(data_path: str, collection_name: str):
    """
    Main function to handle the ingestion process for a given path and collection.

    :param data_path: Path to the directory containing data to ingest.
    :param collection_name: Name of the ChromaDB collection to use.
    """
    print(f"--- Starting ARK Knowledge Ingestion for collection: '{collection_name}' ---")

    # --- 1. Load Documents ---
    print(f"Step 1: Loading documents from '{data_path}'...")
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' not found.")
        return

    # Define a list of file extensions that are likely to contain useful text.
    # Add or remove extensions based on your project's needs.
    # This list now includes source code, markup, and Verilog/SystemVerilog files.
    included_extensions = [
        "*.txt", "*.md", "*.rst",          # Documentation
        "*.py", "*.js", "*.html", "*.css",  # Web & Python
        "*.c", "*.h", "*.cpp", "*.hpp",     # C/C++
        "*.v", "*.sv", "*.svh",             # Verilog/SystemVerilog
        "*.pdf", "*.png", "*.jpg",          # Documents & Images (with Tesseract)
        "Makefile", "*.sh", "*.yml", "*.toml" # Config & Scripts
    ]

    documents = []
    print("Scanning for relevant file types...")
    # Loop through each extension and load the files
    for ext in included_extensions:
        try:
            loader = DirectoryLoader(
                data_path,
                glob=f"**/{ext}",
                recursive=True,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True, # Suppress warnings for individual broken files 
                loader_cls=lambda p: UnstructuredFileLoader(p, mode="single", strategy="fast"),
            )
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading files with extension {ext}: {e}")
            continue

    if not documents:
        print("No documents found to ingest. Exiting.")
        return
    print(f"Successfully loaded {len(documents)} documents.")

    # --- 2. Split Documents into Chunks ---
    print("Step 2: Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks.")

    # --- 3. Initialize Embedding Model ---
    print(f"Step 3: Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')
    print("Embedding model loaded onto GPU.")

    # --- 4. Initialize Vector Database ---
    print(f"Step 4: Initializing vector database at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Now using the collection name passed as an argument
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{collection_name}' is ready.")

    # --- 5. Generate Embeddings and Ingest into DB ---
    print("Step 5: Generating embeddings and ingesting into the database...")
    start_time = time.time()

    # Ingest in batches
    batch_size = 100
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]

        batch_texts = [doc.page_content for doc in batch]
        # Langchain's loader adds 'source' to metadata, which is excellent for citation.
        metadatas = [doc.metadata for doc in batch]
        ids = [f"{collection_name}_{i+j}" for j in range(len(batch))]

        embeddings = embedding_model.encode(batch_texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=metadatas
        )
        print(f"  - Ingested batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")

    end_time = time.time()
    print(f"\nIngestion complete. Took {end_time - start_time:.2f} seconds.")
    print(f"Total documents in collection '{collection_name}': {collection.count()}")
    print(f"--- ARK Knowledge Ingestion for '{collection_name}' Finished ---")

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Ingest data into ARK's knowledge base.")
    parser.add_argument("--path", type=str, required=True, help="The path to the directory of data to ingest.")
    parser.add_argument("--collection", type=str, required=True, help="The name of the ChromaDB collection to use.")

    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(data_path=args.path, collection_name=args.collection)
